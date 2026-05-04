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
    if exist(topk_json, 'file')
        S = jsondecode(fileread(topk_json));
        EC50  = json_float_field(S, 'ec50_hat',  4.0);
        gamma = json_float_field(S, 'gamma_hat', 2.0);
    else
        EC50  = 4.0;
        gamma = 2.0;
    end

    % -------- 读取 NLME 结果（包含 EC50/gamma 列）--------
    R = readtable(simbio_csv);
    R = sortrows(R, 'rank');

    % 如果 simbio_csv 没有 EC50/gamma 列（向后兼容），使用默认值
    has_ec50_gamma = ismember('EC50', R.Properties.VariableNames) && ismember('gamma', R.Properties.VariableNames);

    % -------- 逐候选诊断 --------
    for i = 1:height(R)
        terms_str = string(R.terms{i});
        if ~R.converged(i)
            fprintf('[Rank %d] %s: not converged, skip.\n', i, terms_str);
            continue;
        end

        terms = strsplit(terms_str, ' + ');
        p = numel(terms);

        % 读取该候选的 theta（theta1, theta2, ...）
        theta = zeros(p, 1);
        has_theta = true;
        for k = 1:p
            col = sprintf('theta%d', k);
            if ismember(col, R.Properties.VariableNames)
                theta(k) = R{i, col};
            else
                has_theta = false;
                break
            end
        end
        if ~has_theta
            fprintf('[Rank %d] %s: no theta columns in simbio_csv, skip.\n', i, terms_str);
            continue;
        end

        % 读取该候选的 EC50/gamma（从 simbio_results 表中读取）
        if has_ec50_gamma
            ec50_i = double(R{i, 'EC50'});
            gamma_i = double(R{i, 'gamma'});
        else
            ec50_i = EC50;   % 保持全局默认值
            gamma_i = gamma;
        end

        % 如果仍是默认值（NaN），回退到全局 EC50/gamma
        if isnan(ec50_i), ec50_i = EC50; end
        if isnan(gamma_i), gamma_i = gamma; end

        fprintf('[Rank %d] %s: theta = [%s], EC50 = %.4f, gamma = %.4f\n', ...
            i, terms_str, num2str(theta', '%.4f '), ec50_i, gamma_i);

        % ---- (1) ODE 预测所有 subject ----
        ypred_all = []; yobs_all = []; t_all = []; sid_all = [];
        for s = 1:nSub
            yhat = sim_subject(theta, subj{s}.t, subj{s}.c, subj{s}.y(1), terms, ec50_i, gamma_i);
            m = min(numel(subj{s}.y), numel(yhat));
            ypred_all = [ypred_all; yhat(1:m)];
            yobs_all  = [yobs_all;  subj{s}.y(1:m)];
            t_all     = [t_all;     subj{s}.t(1:m)];
            sid_all   = [sid_all;   repmat(sid_u(s), m, 1)];
        end
        resid = ypred_all - yobs_all;

        % ---- (2) GOF 图 ----
        prefix = sprintf('m%d', i);
        plot_gof(ypred_all, yobs_all, t_all, sid_all, prefix, out_fig_dir);

        % ---- (3) 残差图 ----
        plot_residual(resid, t_all, prefix, out_fig_dir);

        % ---- (4) VPC ----
        plot_vpc(subj, theta, terms, ec50_i, gamma_i, binEdges, prefix, out_fig_dir);

        % ---- (5) Bootstrap ----
        if ~skip_bootstrap
            [n_ok, boot_theta] = bootstrap_refit(subj, terms, theta, ec50_i, gamma_i, 100, 20260420 + i);
            plot_bootstrap(boot_theta, theta, terms, prefix, out_fig_dir);
            fprintf('  -> Boot=%d/100\n', n_ok);
        else
            fprintf('  -> Bootstrap skipped\n');
        end
    end

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
function plot_vpc(subj, theta, terms, EC50, gamma, binEdges, prefix, figDir)
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
            yhat_full = sim_subject(theta, t_full, c_full, R0, terms, EC50, gamma);

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
function [n_ok, boot_theta] = bootstrap_refit(subj, terms, theta0, EC50, gamma, nBoot, seed_)
    rng(seed_);
    nSub = numel(subj);
    p = numel(terms);
    boot_theta = zeros(p, nBoot);
    n_ok = 0;

    for b = 1:nBoot
        bootIdx  = datasample(1:nSub, nSub, 'Replace', true);
        bootSubj = subj(bootIdx);

        obj = @(th) pooled_residual_fast(th, bootSubj, terms, EC50, gamma);

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
function yhat = sim_subject(theta, t, c, R0, terms, EC50, gamma)
    t = t(:); c = c(:);
    [t, ord] = sort(t); c = c(ord);

    [tu, ia] = unique(t, 'stable');
    cu = c(ia);
    if numel(tu) < 2
        yhat = ones(size(t)) * R0;
        return;
    end

    Cfun = @(tt) interp1(tu, cu, tt, 'linear', 'extrap');
    ode  = @(tt, R) rhs_dR(tt, R, Cfun, theta, terms, EC50, gamma);
    [tsol, Rsol] = ode45(ode, [tu(1) tu(end)], R0);
    yhat = interp1(tsol, Rsol(:), t, 'linear', 'extrap');
    yhat = yhat(:);
end


function dR = rhs_dR(t, R, Cfun, theta, terms, EC50, gamma)
    R = R(1);
    C = max(Cfun(t), 1e-10);

    lib = containers.Map('KeyType', 'char', 'ValueType', 'double');
    lib('1')           = 1.0;
    lib('R')           = R;
    lib('C')           = C;
    lib('C^2')         = C^2;
    lib('Emax(C)')     = C / (EC50 + C);
    lib('Hill(C)')     = (C^gamma) / (EC50^gamma + C^gamma);
    lib('C*R')         = C * R;
    lib('Emax(C)*R')   = (C / (EC50 + C)) * R;
    lib('Hill(C)*R')   = ((C^gamma) / (EC50^gamma + C^gamma)) * R;

    dR = 0.0;
    for k = 1:numel(terms)
        key = char(terms{k});
        if isKey(lib, key)
            dR = dR + theta(k) * lib(key);
        end
    end
end


function r = pooled_residual_fast(theta, subj, terms, EC50, gamma)
    r = [];
    for i = 1:numel(subj)
        yhat = sim_subject(theta, subj{i}.t, subj{i}.c, subj{i}.y(1), terms, EC50, gamma);
        m = min(numel(subj{i}.y), numel(yhat));
        r = [r; yhat(1:m) - subj{i}.y(1:m)];
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
