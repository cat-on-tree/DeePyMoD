function diagnostics_plots(data_csv, simbio_csv, topk_json, out_fig_dir, binEdges, skip_bootstrap)
% VPC + Bootstrap 诊断图（使用 Step 5 NLME 拟合结果）
%
% Inputs:
%   data_csv    : pkpd_long.csv (sid,time,C_obs,R_obs)
%   simbio_csv  : simbiology_results.csv (rank,terms,converged,theta1,...,thetak)
%   topk_json   : topk_candidates.json（含 ec50_hat, gamma_hat）
%   out_fig_dir : 图表输出目录
%   binEdges    : 时间分箱边界向量
%   skip_bootstrap : true/false — true时跳过 Bootstrap 节省时间

    if nargin < 5
        binEdges = [0.0 1.0 4.0 8.0 24.0];
    end
    if nargin < 6
        skip_bootstrap = false;
    end

    % Ensure binEdges is a numeric vector (could be passed as cell from Python)
    if iscell(binEdges)
        binEdges = cellfun(@(x) double(x), binEdges);
    end
    binEdges = sort(binEdges(:)');

    if ~exist(out_fig_dir, 'dir')
        mkdir(out_fig_dir);
    end

    % -------- 读取 PK/PD 数据 --------
    T = readtable(data_csv);
    T.Properties.VariableNames = {'sid','time','C_obs','R_obs'};
    sid_u = sort(unique(T.sid));
    nSub = numel(sid_u);

    subj = cell(nSub, 1);
    for i = 1:nSub
        idx = (T.sid == sid_u(i));
        ti = T.time(idx); ci = T.C_obs(idx); yi = T.R_obs(idx);
        [ti, ord] = sort(ti);
        subj{i}.t = ti(:);
        subj{i}.c = ci(ord);
        subj{i}.y = yi(ord);
        subj{i}.sid = sid_u(i);
    end

    % -------- 读取 EC50 / gamma（从 topk_candidates.json）--------
    % ec50_hat / gamma_hat 在 JSON 顶层（payload 级别），不在 candidates 内
    topk_candidates = [];
    if exist(topk_json, 'file')
        S = jsondecode(fileread(topk_json));
        EC50  = json_float_field(S, 'ec50_hat',  4.0);
        gamma = json_float_field(S, 'gamma_hat', 2.0);
        if isstruct(S) && isfield(S, 'candidates') && ~isempty(S.candidates)
            topk_candidates = S.candidates;
        end
    else
        EC50  = 4.0;
        gamma = 2.0;
    end

    % -------- 读取 NLME 结果（包含 EC50/gamma 列）--------
    R = readtable(simbio_csv);
    R = sortrows(R, 'rank');

    % 如果 simbio_csv 没有 EC50/gamma 列（向后兼容），使用默认值
    has_ec50_gamma = ismember('EC50', R.Properties.VariableNames) && ismember('gamma', R.Properties.VariableNames);

    mech_rank = nan(height(R), 1);
    mech_converged = false(height(R), 1);
    mech_has_tmdd = false(height(R), 1);
    mech_gamma_lb = false(height(R), 1);
    mech_ec50_bound = false(height(R), 1);
    mech_neg_state_ratio = nan(height(R), 1);
    mech_state_explosion = false(height(R), 1);
    mech_residual_bias = nan(height(R), 1);

    % -------- 逐候选诊断 --------
    for i = 1:height(R)
        rank_i = i;
        if ismember('rank', R.Properties.VariableNames)
            rank_i = double(R.rank(i));
            if ~isfinite(rank_i), rank_i = i; end
        end
        mech_rank(i) = rank_i;
        mech_converged(i) = logical(R.converged(i));
        try
            terms_str = string(R.terms{i});
            raw_terms = terms_str;
            if ~isempty(topk_candidates)
                cand = get_candidate_by_rank(topk_candidates, rank_i);
                if isstruct(cand) && isfield(cand, 'terms') && ~isempty(cand.terms)
                    raw_terms = cand.terms;
                end
            end
            [eq_terms, flat_terms, terms_label] = parse_candidate_terms(raw_terms);
            has_tmdd_terms = candidate_has_tmdd_terms(flat_terms);
            mech_has_tmdd(i) = has_tmdd_terms;

            if has_ec50_gamma
                ec50_i = double(R{i, 'EC50'});
                gamma_i = double(R{i, 'gamma'});
            else
                ec50_i = EC50;
                gamma_i = gamma;
            end
            if isnan(ec50_i), ec50_i = EC50; end
            if isnan(gamma_i), gamma_i = gamma; end
            mech_gamma_lb(i) = (gamma_i <= 1.05);
            mech_ec50_bound(i) = (ec50_i <= 0.55) || (ec50_i >= 19.5);

            if ~R.converged(i)
                fprintf('[Rank %d] %s: not converged, skip.\n', rank_i, terms_label);
                continue;
            end

            p = numel(flat_terms);
            if p == 0
                fprintf('[Rank %d] %s: parsed empty terms, skip.\n', rank_i, terms_str);
                continue;
            end

            % 读取该候选的 theta（theta1, theta2, ...）
            theta = zeros(p, 1);
            missing_cols = {};
            for k = 1:p
                col = sprintf('theta%d', k);
                if ismember(col, R.Properties.VariableNames)
                    theta(k) = double(R{i, col});
                else
                    missing_cols{end+1} = col; %#ok<AGROW>
                end
            end
            if ~isempty(missing_cols)
                fprintf('[Rank %d] %s: missing theta columns (%s) for p=%d, skip.\n', ...
                    rank_i, terms_label, strjoin(missing_cols, ', '), p);
                continue;
            end

            fprintf('[Rank %d] %s: theta = [%s], EC50 = %.4f, gamma = %.4f\n', ...
                rank_i, terms_label, num2str(theta', '%.4f '), ec50_i, gamma_i);

            % ---- (1) ODE 预测所有 subject ----
            all_states = [];
            ypred_all = []; yobs_all = []; t_all = []; sid_all = [];
            for s = 1:nSub
                [yhat, states_hat] = simulate_subject_dispatch(theta, subj{s}.t, subj{s}.c, subj{s}.y(1), eq_terms, ec50_i, gamma_i);
                m = min(numel(subj{s}.y), numel(yhat));
                ypred_all = [ypred_all; yhat(1:m)];
                yobs_all  = [yobs_all;  subj{s}.y(1:m)];
                t_all     = [t_all;     subj{s}.t(1:m)];
                sid_all   = [sid_all;   repmat(sid_u(s), m, 1)];
                if ~isempty(states_hat)
                    all_states = [all_states; states_hat(1:m, :)]; %#ok<AGROW>
                end
            end
            resid = ypred_all - yobs_all;
            mech_residual_bias(i) = mean(resid);
            if ~isempty(all_states)
                valid_state = isfinite(all_states);
                neg_mask = valid_state & (all_states < -1e-6);
                denom = sum(valid_state(:));
                if denom > 0
                    mech_neg_state_ratio(i) = sum(neg_mask(:)) / denom;
                end
                mech_state_explosion(i) = any(valid_state(:) & (abs(all_states(:)) > 1e4));
            end

            % ---- (2) GOF 图 ----
            prefix = sprintf('m%d', i);
            plot_gof(ypred_all, yobs_all, t_all, sid_all, prefix, out_fig_dir);

            % ---- (3) 残差图 ----
            plot_residual(resid, t_all, prefix, out_fig_dir);

            % ---- (4) VPC ----
            plot_vpc(subj, theta, eq_terms, ec50_i, gamma_i, binEdges, prefix, out_fig_dir);

            % ---- (5) Bootstrap ----
            if ~skip_bootstrap
                [n_ok, boot_theta] = bootstrap_refit(subj, eq_terms, theta, ec50_i, gamma_i, 100, 20260420 + i);
                plot_bootstrap(boot_theta, theta, flat_terms, prefix, out_fig_dir);
                fprintf('  -> Boot=%d/100\n', n_ok);
            else
                fprintf('  -> Bootstrap skipped\n');
            end
        catch ME
            err_file = fullfile(out_fig_dir, sprintf('m%d_error.txt', i));
            fid = fopen(err_file, 'w');
            if fid ~= -1
                fprintf(fid, '[Rank %d] Diagnostics error: %s\n\n', rank_i, ME.message);
                fprintf(fid, '%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
                fclose(fid);
            end
            fprintf('[Rank %d] diagnostics error, skipped. See %s\n', rank_i, err_file);
            continue;
        end
    end

    mech_tbl = table( ...
        mech_rank, mech_converged, mech_has_tmdd, mech_gamma_lb, mech_ec50_bound, ...
        mech_neg_state_ratio, mech_state_explosion, mech_residual_bias, ...
        'VariableNames', {'rank','converged','has_tmdd_terms','gamma_at_lower_bound','ec50_at_bound', ...
                          'neg_state_ratio','state_explosion_flag','residual_bias'});
    writetable(mech_tbl, fullfile(out_fig_dir, 'mechanism_checks.csv'));

    fprintf('Done. Figures in %s\n', out_fig_dir);
end


% ==============================================================================
% GOF 图
% ==============================================================================
function plot_gof(ypred, yobs, t, sid, prefix, figDir)
    fig = figure('Visible', 'off', 'Position', [100 100 1200 400]);

    ax1 = subplot(1,3,1);
    scatter(ypred, yobs, 12, 'filled', 'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'none');
    hold on;
    lims = [min([ypred; yobs])*0.95, max([ypred; yobs])*1.05];
    plot(lims, lims, 'r--', 'LineWidth', 1.5);
    xlabel('Predicted R'); ylabel('Observed R');
    title(['GOF [' prefix ']']); grid on;

    ax2 = subplot(1,3,2);
    sids = unique(sid);
    cmap = lines(numel(sids));
    for j = 1:numel(sids)
        mask = (sid == sids(j));
        scatter(ypred(mask), yobs(mask), 10, 'filled', ...
            'MarkerFaceColor', cmap(j,:), 'MarkerFaceAlpha', 0.7);
        hold on;
    end
    plot(lims, lims, 'r--', 'LineWidth', 1.5);
    xlabel('Predicted R'); ylabel('Observed R');
    title(['GOF by subject [' prefix ']']); grid on;

    ax3 = subplot(1,3,3);
    scatter(t, ypred - yobs, 10, 'filled', 'MarkerFaceAlpha', 0.5, ...
        'MarkerEdgeColor', 'none', 'MarkerFaceColor', [0.85 0.45 0.3]);
    hold on;
    plot([min(t) max(t)], [0 0], 'k--', 'LineWidth', 1.2);
    xlabel('time'); ylabel('Residual');
    title(['Residuals vs Time [' prefix ']']); grid on;

    print(fig, fullfile(figDir, [prefix '_gof.png']), '-dpng', '-r160');
    close(fig);
end


% ==============================================================================
% 残差图
% ==============================================================================
function plot_residual(resid, t, prefix, figDir)
    fig = figure('Visible', 'off', 'Position', [100 100 800 350]);

    ax1 = subplot(1,2,1);
    qqplot(resid);
    h = get(ax1, 'Children');
    if numel(h) >= 1 && isprop(h(1), 'SizeData')
        try set(h(1), 'MarkerFaceColor', [0.4 0.6 0.8], 'SizeData', 20); end
    end
    title(['Normal QQ [' prefix ']']); grid on;

    ax2 = subplot(1,2,2);
    histogram(resid, 25, 'Normalization', 'pdf', ...
        'FaceColor', [0.85 0.55 0.3], 'EdgeColor', 'white');
    hold on;
    xmin = min(resid); xmax = max(resid);
    if xmax > xmin
        xs = linspace(xmin, xmax, 200);
        plot(xs, normpdf(xs, mean(resid), std(resid)), 'b-', 'LineWidth', 2);
    end
    xlabel('Residual'); ylabel('Density');
    mu = mean(resid); sg = std(resid);
    title(sprintf('Residual Dist [%s] (\\mu=%.2f, \\sigma=%.2f)', prefix, mu, sg));
    grid on;

    print(fig, fullfile(figDir, [prefix '_residual.png']), '-dpng', '-r160');
    close(fig);
end


% ==============================================================================
% VPC：对每个 subject 从 t=0 积分完整轨迹，再按 bin 切片取分位数
% ==============================================================================
function plot_vpc(subj, theta, eq_terms, EC50, gamma, binEdges, prefix, figDir)
    nSub = numel(subj);
    nBin = numel(binEdges) - 1;

    % ---- 观测分位数（每个 bin）----
    obs_lo = zeros(nBin,1); obs_me = zeros(nBin,1); obs_hi = zeros(nBin,1);
    tMid   = zeros(nBin,1);

    for b = 1:nBin
        tLo = binEdges(b); tHi = binEdges(b+1);
        binObs = [];
        for s = 1:nSub
            mask = (subj{s}.t >= tLo & subj{s}.t < tHi);
            if any(mask)
                binObs = [binObs; subj{s}.y(mask)];
            end
        end
        tMid(b) = mean([tLo tHi]);
        if ~isempty(binObs)
            obs_lo(b) = prctile(binObs, 5);
            obs_me(b) = prctile(binObs, 50);
            obs_hi(b) = prctile(binObs, 95);
        end
    end

    % ---- 模型预测：每个 subject 从 t=0 积分到 max(time)，再按 bin 切片 ----
    mod_lo = zeros(nBin,1); mod_me = zeros(nBin,1); mod_hi = zeros(nBin,1);

    for b = 1:nBin
        tLo = binEdges(b); tHi = binEdges(b+1);
        binPreds = [];

        for s = 1:nSub
            t_full = subj{s}.t;
            c_full = subj{s}.c;
            R0     = subj{s}.y(1);   % t=0 的观测值作为初始条件

            % 积分完整轨迹
            yhat_full = simulate_subject_dispatch(theta, t_full, c_full, R0, eq_terms, EC50, gamma);

            % 提取 bin 内预测
            mask = (t_full >= tLo & t_full < tHi);
            if any(mask)
                binPreds = [binPreds; yhat_full(mask)];
            end
        end

        if ~isempty(binPreds)
            mod_lo(b) = prctile(binPreds, 5);
            mod_me(b) = prctile(binPreds, 50);
            mod_hi(b) = prctile(binPreds, 95);
        end
    end

    % ---- 绘图 ----
    fig = figure('Visible', 'off', 'Position', [100 100 700 420]);

    % 观测 PI
    fill([tMid(:); flipud(tMid(:))], [obs_lo(:); flipud(obs_hi(:))], ...
        [0.3 0.65 1.0], 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    hold on;
    plot(tMid, obs_me, 'b-', 'LineWidth', 2, 'DisplayName', 'Obs median');
    plot(tMid, obs_lo, 'b--', 'LineWidth', 1); plot(tMid, obs_hi, 'b--', 'LineWidth', 1);

    % 模型 PI
    fill([tMid(:); flipud(tMid(:))], [mod_lo(:); flipud(mod_hi(:))], ...
        [1.0 0.35 0.35], 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    plot(tMid, mod_me, 'r-', 'LineWidth', 2, 'DisplayName', 'Model median');
    plot(tMid, mod_lo, 'r--', 'LineWidth', 1); plot(tMid, mod_hi, 'r--', 'LineWidth', 1);

    xlabel('Time'); ylabel('R_obs');
    title(['VPC [' prefix ']']); legend('Location', 'best'); grid on;
    print(fig, fullfile(figDir, [prefix '_vpc.png']), '-dpng', '-r160');
    close(fig);
end


% ==============================================================================
% Bootstrap
% ==============================================================================
function [n_ok, boot_theta] = bootstrap_refit(subj, eq_terms, theta0, EC50, gamma, nBoot, seed_)
    rng(seed_);
    nSub = numel(subj);
    flat_terms = flatten_eq_terms(eq_terms);
    p = numel(flat_terms);
    boot_theta = zeros(p, nBoot);
    n_ok = 0;

    for b = 1:nBoot
        bootIdx  = datasample(1:nSub, nSub, 'Replace', true);
        bootSubj = subj(bootIdx);

        obj = @(th) pooled_residual_fast(th, bootSubj, eq_terms, EC50, gamma);

        lb = -50 * ones(p, 1);
        ub =  50 * ones(p, 1);
        opts = optimoptions('lsqnonlin', 'Display', 'off', ...
            'MaxIterations', 150, 'FunctionTolerance', 1e-5, 'StepTolerance', 1e-5);

        try
            [th_hat, ~, ~, exitflag] = lsqnonlin(obj, theta0, lb, ub, opts);
            if exitflag > 0
                boot_theta(:, b) = th_hat;
                n_ok = n_ok + 1;
            end
        catch
            % skip
        end
    end
end


% ==============================================================================
% Bootstrap 直方图
% ==============================================================================
function plot_bootstrap(boot_theta, theta_fit, terms, prefix, figDir)
    [p, ~] = size(boot_theta);
    n_ok = sum(any(boot_theta ~= 0, 1));

    fig = figure('Visible', 'off', 'Position', [100 100 300*p 300]);
    for k = 1:p
        vals = boot_theta(k, :);
        vals = vals(vals ~= 0);

        subplot(1, p, k);
        if ~isempty(vals)
            histogram(vals, 25, 'Normalization', 'pdf', ...
                'FaceColor', [0.4 0.6 0.9], 'EdgeColor', 'white');
            hold on;
            xline(theta_fit(k), 'r-', 'LineWidth', 2, ...
                'Label', sprintf('Est=%.3f', theta_fit(k)));
            q2  = prctile(vals, 2.5);
            q97 = prctile(vals, 97.5);
            xline(q2,  'LineStyle', '--', 'Color', [1 0.5 0], 'LineWidth', 1.5, 'Label', sprintf('2.5%%=%.3f', q2));
            xline(q97, 'LineStyle', '--', 'Color', [1 0.5 0], 'LineWidth', 1.5, 'Label', sprintf('97.5%%=%.3f', q97));
        else
            text(0.5, 0.5, sprintf('No convergence\n(theta=%.3f)', theta_fit(k)), ...
                'Units', 'normalized', 'HorizontalAlignment', 'center');
        end
        xlabel(sprintf('\\theta[%s]', terms{k}));
        ylabel('Density');
        title(sprintf('Bootstrap: %s [%s]', terms{k}, prefix));
        grid on;
    end
    print(fig, fullfile(figDir, [prefix '_bootstrap.png']), '-dpng', '-r160');
    close(fig);
end


% ==============================================================================
% ODE 积分
% ==============================================================================
function [yhat, state_hat] = simulate_subject_dispatch(theta, t, c, R0, eq_terms, EC50, gamma)
    if numel(eq_terms) > 1
        [yhat, state_hat] = simulate_subject_multi(theta, t, c, R0, eq_terms, EC50, gamma);
    else
        [yhat, state_hat] = simulate_subject_single(theta, t, c, R0, eq_terms{1}, EC50, gamma);
    end
end


function [yhat, state_hat] = simulate_subject_single(theta, t, c, R0, terms, EC50, gamma)
    t = t(:); c = c(:);
    [t, ord] = sort(t); c = c(ord);

    [tu, ia] = unique(t, 'stable');
    cu = c(ia);
    if numel(tu) < 2
        yhat = ones(size(t)) * R0;
        state_hat = [yhat, nan(size(yhat)), nan(size(yhat))];
        return;
    end

    Cfun = @(tt) interp1(tu, cu, tt, 'linear', 'extrap');
    ode = @(tt, R) rhs_single(tt, R, Cfun, theta, terms, EC50, gamma);
    [tsol, Rsol] = ode45(ode, [tu(1) tu(end)], R0);
    yhat = interp1(tsol, Rsol(:), t, 'linear', 'extrap');
    yhat = yhat(:);
    state_hat = [yhat, nan(size(yhat)), nan(size(yhat))];
end


function [yhat, state_hat] = simulate_subject_multi(theta, t, c, R0, eq_terms, EC50, gamma)
    t = t(:); c = c(:);
    [t, ord] = sort(t); c = c(ord);

    [tu, ia] = unique(t, 'stable');
    cu = c(ia);
    if numel(tu) < 2
        yhat = ones(size(t)) * R0;
        state_hat = [yhat, nan(size(yhat)), nan(size(yhat))];
        return;
    end

    Cfun = @(tt) interp1(tu, cu, tt, 'linear', 'extrap');
    nState = min(max(numel(eq_terms), 1), 3);
    x0 = zeros(nState, 1);
    x0(1) = R0;

    ode = @(tt, X) rhs_multi(tt, X, Cfun, theta, eq_terms, EC50, gamma);
    [tsol, Xsol] = ode45(ode, [tu(1) tu(end)], x0);
    yhat = interp1(tsol, Xsol(:,1), t, 'linear', 'extrap');
    yhat = yhat(:);
    state_hat = nan(numel(t), 3);
    state_hat(:, 1) = yhat;
    for iState = 2:min(size(Xsol,2), 3)
        state_hat(:, iState) = interp1(tsol, Xsol(:, iState), t, 'linear', 'extrap');
    end
end


function dR = rhs_single(t, R, Cfun, theta, terms, EC50, gamma)
    c_now = Cfun(t);
    ctx.t = t;
    ctx.R = R(1);
    ctx.C = c_now(1);
    ctx.CpR = 0.0;
    ctx.Ct = 0.0;
    dR = sum_terms(theta, terms, ctx, EC50, gamma);
end


function dX = rhs_multi(t, X, Cfun, theta, eq_terms, EC50, gamma)
    nState = min(max(numel(eq_terms), 1), 3);
    dX = zeros(nState, 1);
    c_now = Cfun(t); c_now = c_now(1);
    offset = 0;

    for iEq = 1:nState
        ctx.t = t;
        ctx.R = X(1);
        ctx.C = c_now;
        ctx.CpR = 0.0;
        ctx.Ct = 0.0;
        if nState >= 2, ctx.CpR = X(2); end
        if nState >= 3, ctx.Ct = X(3); end

        eq_i = eq_terms{iEq};
        ni = numel(eq_i);
        if ni > 0
            i1 = offset + 1;
            i2 = min(offset + ni, numel(theta));
            if i2 >= i1
                theta_i = theta(i1:i2);
                dX(iEq) = sum_terms(theta_i, eq_i, ctx, EC50, gamma);
            end
            offset = offset + ni;
        end
    end
end


function v = sum_terms(theta, terms, ctx, EC50, gamma)
    v = 0.0;
    n = min(numel(theta), numel(terms));
    for k = 1:n
        v = v + theta(k) * evaluate_term(terms{k}, ctx, EC50, gamma);
    end
end


function v = evaluate_term(term, ctx, EC50, gamma)
    key = lower(strrep(strtrim(char(string(term))), ' ', ''));
    switch key
        case '1'
            v = 1.0;
        case 'r'
            v = ctx.R;
        case 'c'
            v = ctx.C;
        case 'c^2'
            v = ctx.C^2;
        case 'emax(c)'
            v = safe_emax(ctx.C, EC50);
        case 'hill(c)'
            v = safe_hill(ctx.C, EC50, gamma);
        case 'c*r'
            v = ctx.C * ctx.R;
        case 'emax(c)*r'
            v = safe_emax(ctx.C, EC50) * ctx.R;
        case 'hill(c)*r'
            v = safe_hill(ctx.C, EC50, gamma) * ctx.R;
        case 'cpr'
            v = ctx.CpR;
        case 'cpr*r'
            v = ctx.CpR * ctx.R;
        case 'ct'
            v = ctx.Ct;
        case 'c-ct'
            v = ctx.C - ctx.Ct;
        case 'ct*r'
            v = ctx.Ct * ctx.R;
        case 'cos(2pi*t/24)'
            v = cos(2*pi*ctx.t/24);
        case 'sin(2pi*t/24)'
            v = sin(2*pi*ctx.t/24);
        otherwise
            tok = regexp(key, '^emax\(([^)]+)\)$', 'tokens', 'once');
            if ~isempty(tok)
                x = eval_base_symbol(tok{1}, ctx);
                if isnan(x), v = 0.0; else, v = safe_emax(x, EC50); end
                return;
            end

            tok = regexp(key, '^hill\(([^)]+)\)$', 'tokens', 'once');
            if ~isempty(tok)
                x = eval_base_symbol(tok{1}, ctx);
                if isnan(x), v = 0.0; else, v = safe_hill(x, EC50, gamma); end
                return;
            end

            tok = regexp(key, '^([a-z][a-z0-9\-]*)\*r$', 'tokens', 'once');
            if ~isempty(tok)
                x = eval_base_symbol(tok{1}, ctx);
                if isnan(x), v = 0.0; else, v = x * ctx.R; end
                return;
            end

            v = 0.0;
    end
end


function x = eval_base_symbol(symb, ctx)
    switch lower(strrep(char(string(symb)), ' ', ''))
        case 'c'
            x = ctx.C;
        case 'r'
            x = ctx.R;
        case 'cpr'
            x = ctx.CpR;
        case 'ct'
            x = ctx.Ct;
        case 'c-ct'
            x = ctx.C - ctx.Ct;
        otherwise
            x = NaN;
    end
end


function v = safe_emax(x, EC50)
    den = EC50 + x;
    if abs(den) < eps
        v = 0.0;
    else
        v = x / den;
    end
end


function v = safe_hill(x, EC50, gamma)
    x = max(x, 0.0);
    den = EC50^gamma + x^gamma;
    if den <= 0
        v = 0.0;
    else
        v = x^gamma / den;
    end
end


function r = pooled_residual_fast(theta, subj, eq_terms, EC50, gamma)
    r = [];
    for i = 1:numel(subj)
        yhat = simulate_subject_dispatch(theta, subj{i}.t, subj{i}.c, subj{i}.y(1), eq_terms, EC50, gamma);
        m = min(numel(subj{i}.y), numel(yhat));
        r = [r; yhat(1:m) - subj{i}.y(1:m)];
    end
end


function [eq_terms, flat_terms, terms_label] = parse_candidate_terms(raw_terms)
    if iscell(raw_terms) && ~isempty(raw_terms)
        is_single_eq = all(cellfun(@is_text_term, raw_terms));
        if is_single_eq
            terms = to_cellstr_terms(raw_terms);
            if numel(terms) == 1
                [eq_terms, flat_terms, terms_label] = parse_terms_label(terms{1});
            else
                eq_terms = {terms};
                flat_terms = terms;
                terms_label = strjoin(flat_terms, ' + ');
            end
            return;
        end

        nEq = min(numel(raw_terms), 3);
        eq_terms = cell(1, nEq);
        eq_labels = cell(1, nEq);
        flat_terms = {};
        for iEq = 1:nEq
            eq_terms{iEq} = to_cellstr_terms(raw_terms{iEq});
            flat_terms = [flat_terms, eq_terms{iEq}]; %#ok<AGROW>
            eq_labels{iEq} = sprintf('Eq%d: %s', iEq, strjoin(eq_terms{iEq}, ' + '));
        end
        terms_label = strjoin(eq_labels, ' || ');
        return;
    end

    [eq_terms, flat_terms, terms_label] = parse_terms_label(raw_terms);
end


function [eq_terms, flat_terms, terms_label] = parse_terms_label(label)
    txt = strtrim(char(string(label)));
    if isempty(txt)
        eq_terms = {{}};
        flat_terms = {};
        terms_label = '';
        return;
    end

    has_multi = contains(txt, '||') || ~isempty(regexp(txt, 'Eq\d+\s*:', 'once'));
    if has_multi
        chunks = regexp(txt, '\|\|', 'split');
        eq_terms = {};
        for iEq = 1:numel(chunks)
            seg = strtrim(chunks{iEq});
            seg = regexprep(seg, '^Eq\d+\s*:\s*', '');
            terms_i = split_terms(seg);
            if ~isempty(terms_i)
                eq_terms{end+1} = terms_i; %#ok<AGROW>
            end
        end
        if isempty(eq_terms)
            eq_terms = {split_terms(txt)};
        end
    else
        eq_terms = {split_terms(txt)};
    end

    flat_terms = flatten_eq_terms(eq_terms);
    if numel(eq_terms) > 1
        eq_labels = cell(1, numel(eq_terms));
        for iEq = 1:numel(eq_terms)
            eq_labels{iEq} = sprintf('Eq%d: %s', iEq, strjoin(eq_terms{iEq}, ' + '));
        end
        terms_label = strjoin(eq_labels, ' || ');
    else
        terms_label = strjoin(flat_terms, ' + ');
    end
end


function terms = split_terms(seg)
    parts = regexp(strtrim(seg), '\s*\+\s*', 'split');
    parts = parts(~cellfun(@isempty, parts));
    terms = to_cellstr_terms(parts);
end


function tf = is_text_term(x)
    tf = ischar(x) || (isstring(x) && isscalar(x));
end


function terms = to_cellstr_terms(x)
    if ischar(x)
        terms = {x};
    elseif isstring(x)
        terms = cellstr(x(:));
    elseif iscell(x)
        terms = cell(size(x));
        for i = 1:numel(x)
            xi = x{i};
            if ischar(xi)
                terms{i} = xi;
            elseif isstring(xi)
                if isscalar(xi)
                    terms{i} = char(xi);
                else
                    tmp = cellstr(xi(:));
                    terms{i} = strjoin(tmp, ' ');
                end
            else
                terms{i} = char(string(xi));
            end
        end
    else
        terms = {char(string(x))};
    end
    terms = reshape(terms, 1, []);
end


function flat = flatten_eq_terms(eq_terms)
    flat = {};
    for iEq = 1:numel(eq_terms)
        flat = [flat, eq_terms{iEq}]; %#ok<AGROW>
    end
end


function tf = candidate_has_tmdd_terms(flat_terms)
    tf = false;
    for k = 1:numel(flat_terms)
        key = lower(strrep(strtrim(char(string(flat_terms{k}))), ' ', ''));
        if contains(key, 'cpr') || strcmp(key, 'ct') || strcmp(key, 'c-ct')
            tf = true;
            return;
        end
    end
end


function cand = get_candidate_by_rank(candidates, rank_)
    cand = [];
    if ~isstruct(candidates) || isempty(candidates)
        return;
    end
    for j = 1:numel(candidates)
        rj = cand_rank(candidates(j), j);
        if abs(double(rj) - double(rank_)) < 1e-9
            cand = candidates(j);
            return;
        end
    end
end


function r = cand_rank(cand, fallback_rank)
    r = fallback_rank;
    if ~(isstruct(cand) && isfield(cand, 'rank'))
        return;
    end
    val = cand.rank;
    if iscell(val) && ~isempty(val), val = val{1}; end
    if isnumeric(val) && isscalar(val) && isfinite(val)
        r = double(val);
        return;
    end
    if ischar(val) || (isstring(val) && isscalar(val))
        vv = str2double(string(val));
        if ~isnan(vv), r = vv; end
    end
end


% ==============================================================================
% 工具：从 JSON decode 得到的 struct 中读取浮点字段
% 支持顶层或 candidates{1} 内两种位置
% ==============================================================================
function v = json_float_field(S, field, default_)
    v = default_;
    % 尝试顶层
    if isstruct(S) && isfield(S, field)
        val = S.(field);
        if ~isempty(val)
            if iscell(val), val = val{1}; end
            v = max(double(val), eps);
            return;
        end
    end
    % 尝试 candidates{1} 内
    if isstruct(S) && isfield(S, 'candidates') && ~isempty(S.candidates)
        cand = S.candidates(1);
        if isfield(cand, field)
            val = cand.(field);
            if ~isempty(val)
                if iscell(val), val = val{1}; end
                v = max(double(val), eps);
                return;
            end
        end
    end
end
