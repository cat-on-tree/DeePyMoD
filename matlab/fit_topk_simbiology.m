function results = fit_topk_simbiology(data_csv, cand_json, out_csv)
% Robust candidate-wise PKPD fitting (SimBiology installed, no fitproblem dependency)
% NOTE: Uses ODE + lsqnonlin backend for stability across MATLAB versions.
%
% Inputs:
%   data_csv  : pkpd_long.csv with columns sid,time,C_obs,R_obs
%   cand_json : topk_candidates.json (with candidates(k).rank, .terms)
%   out_csv   : output csv path
%
% Output table columns:
%   rank, terms, converged, logLik, AIC, BIC, RMSE, SSE, message, theta1, ..., thetaK, EC50, gamma

    if nargin < 3
        out_csv = 'artifacts/nlme/simbiology_results.csv';
    end

    T = readtable(data_csv);
    T.Properties.VariableNames = {'sid','time','C_obs','R_obs'};
    S = jsondecode(fileread(cand_json));

    sid_u = unique(T.sid);
    nSub = numel(sid_u);

    subj = cell(nSub,1);
    for i = 1:nSub
        idx = (T.sid == sid_u(i));
        ti = T.time(idx);
        ci = T.C_obs(idx);
        yi = T.R_obs(idx);

        [ti, ord] = sort(ti);
        ci = ci(ord);
        yi = yi(ord);

        subj{i}.t = ti(:);
        subj{i}.c = ci(:);
        subj{i}.y = yi(:);
    end

    nCand = numel(S.candidates);

    % ---- 第一遍：确定所有候选的最大维数 ----
    % 包括结构项 theta，以及可选的 EC50/gamma
    max_k = 0;
    any_hill_model = false;
    for j = 1:nCand
        raw_terms = S.candidates(j).terms;
        [~, flat_terms, ~] = parse_candidate_terms(raw_terms);
        p = numel(flat_terms);
        has_hill = any(cellfun(@(x) contains(lower(string(x)), "hill("), flat_terms));
        if has_hill
            any_hill_model = true;
        end
        max_k = max(max_k, p);
    end

    % ---- 预分配 table（统一所有候选的列）----
    % Columns: rank, terms, converged, logLik, AIC, BIC, RMSE, SSE, message,
    %          theta1..thetaK, [EC50, gamma] (only if any_hill_model)
    nVar = 9 + max_k + 2*double(any_hill_model);  % 9 fixed + max_k thetas + [EC50,gamma]
    varNames = cell(1, nVar);
    varNames{1} = 'rank';
    varNames{2} = 'terms';
    varNames{3} = 'converged';
    varNames{4} = 'logLik';
    varNames{5} = 'AIC';
    varNames{6} = 'BIC';
    varNames{7} = 'RMSE';
    varNames{8} = 'SSE';
    varNames{9} = 'message';
    for k = 1:max_k
        varNames{9+k} = sprintf('theta%d', k);
    end
    if any_hill_model
        varNames{nVar-1} = 'EC50';
        varNames{nVar}   = 'gamma';
    end

    varTypes = cell(1, nVar);
    varTypes{1} = 'double';
    varTypes{2} = 'string';
    varTypes{3} = 'logical';
    for v = 4:8
        varTypes{v} = 'double';
    end
    varTypes{9} = 'string';
    for v = 10:nVar
        varTypes{v} = 'double';
    end

    results = table('Size', [0 nVar], 'VariableTypes', varTypes, 'VariableNames', varNames);

    % ---- 逐候选拟合 ----
    for j = 1:nCand
        cand = S.candidates(j);
        raw_terms = cand.terms;
        [eq_terms, flat_terms, terms_label] = parse_candidate_terms(raw_terms);
        p = numel(flat_terms);
        is_multi_eq = numel(eq_terms) > 1;

        has_hill = any(cellfun(@(x) contains(lower(string(x)), "hill("), flat_terms));

        % 如果有 Hill(C) 项，EC50/gamma 也参与拟合
        if has_hill
            % theta: [theta1..thetap, EC50, gamma]
            n_param = p + 2;
            has_EC50_gamma = true;
        else
            n_param = p;
            has_EC50_gamma = false;
        end

        try
            if has_EC50_gamma
                % 从 Python Step 4 结果读取 EC50/gamma 初值（cand_json 中每候选都有）
                ec50_0 = json_float_field(cand, 'ec50_hat', 4.0);
                gamma_0 = json_float_field(cand, 'gamma_hat', 2.0);

                % 尝试读取 Step 4 计算的 theta_hat 作为多起点初值
                % 注意：直接读取，不走 json_float_field（它会对每个元素执行 max(v,eps)，会错误 floor 接近 0 的 theta 值）
                theta_arr = [];
                if isstruct(cand) && isfield(cand, 'theta_hat')
                    val = cand.theta_hat;
                    if isnumeric(val), theta_arr = double(val(:)); end
                end
                if ~isempty(theta_arr) && numel(theta_arr) == p
                    theta0 = [theta_arr; ec50_0; gamma_0];
                else
                    theta0 = [zeros(p,1); ec50_0; gamma_0];
                end

                lb = [-50*ones(p,1); 0.5; 1.0];
                ub = [ 50*ones(p,1); 20.0; 6.0];

                if is_multi_eq
                    obj = @(th) pooled_residual_multi_with_hill(th, subj, eq_terms);
                else
                    obj = @(th) pooled_residual_with_hill(th, subj, flat_terms);
                end
            else
                theta_arr = [];
                if isstruct(cand) && isfield(cand, 'theta_hat')
                    val = cand.theta_hat;
                    if isnumeric(val), theta_arr = double(val(:)); end
                end
                if ~isempty(theta_arr) && numel(theta_arr) == p
                    theta0 = theta_arr;
                else
                    theta0 = zeros(p,1);
                end
                lb = -50*ones(p,1);
                ub =  50*ones(p,1);
                if is_multi_eq
                    obj = @(th) pooled_residual_multi(th, subj, eq_terms);
                else
                    obj = @(th) pooled_residual(th, subj, flat_terms);
                end
            end

            opts = optimoptions('lsqnonlin', ...
                'Display', 'off', ...
                'MaxIterations', 500, ...
                'FunctionTolerance', 1e-9, ...
                'StepTolerance', 1e-9);

            [theta_hat, ~, resid, exitflag] = lsqnonlin(obj, theta0, lb, ub, opts); %#ok<ASGLU>

            n = numel(resid);
            sse = sum(resid.^2);
            rmse = sqrt(sse / max(n,1));

            sigma2 = max(sse / max(n,1), eps);
            logLik = -0.5 * n * (log(2*pi*sigma2) + 1);

            k_eff = numel(theta_hat);
            AIC = -2*logLik + 2*k_eff;
            BIC = -2*logLik + log(max(n,1))*k_eff;

            % Build row: pad theta to max_k with NaN, append EC50/gamma only if hill model
            rowCell = cell(1, nVar);
            rowCell{1} = double(cand_rank(cand, j));
            rowCell{2} = terms_label;
            rowCell{3} = (exitflag > 0);
            rowCell{4} = logLik;
            rowCell{5} = AIC;
            rowCell{6} = BIC;
            rowCell{7} = rmse;
            rowCell{8} = sse;
            rowCell{9} = '';
            for v = 10:nVar
                rowCell{v} = NaN;
            end
            for k = 1:p
                rowCell{9+k} = theta_hat(k);
            end
            if has_EC50_gamma && any_hill_model
                rowCell{9+p+1} = theta_hat(p+1);  % EC50
                rowCell{9+p+2} = theta_hat(p+2);  % gamma
            end
            % else: no EC50/gamma columns in table for non-hill models

            results = [results; rowCell]; %#ok<AGROW>

        catch ME
            errMsg = string(ME.identifier) + " | " + ME.message;
            rowCell = cell(1, nVar);
            rowCell{1} = double(cand_rank(cand, j));
            rowCell{2} = terms_label;
            rowCell{3} = false;
            rowCell{4} = NaN;
            rowCell{5} = NaN;
            rowCell{6} = NaN;
            rowCell{7} = NaN;
            rowCell{8} = NaN;
            rowCell{9} = errMsg;
            for v = 10:nVar
                rowCell{v} = NaN;
            end
            % EC50/gamma left as NaN for non-hill models (columns don't exist)
            results = [results; rowCell]; %#ok<AGROW>
        end
    end

    % ---- 写出 CSV ----
    out_dir = fileparts(out_csv);
    if ~isempty(out_dir) && ~exist(out_dir,'dir')
        mkdir(out_dir);
    end
    writetable(results, out_csv, 'WriteVariableNames', true);
end


% ==============================================================================
% Without Hill(C) terms: standard pooled residual
% ==============================================================================
function r = pooled_residual(theta, subj, terms)
    r = [];
    for i = 1:numel(subj)
        t = subj{i}.t;
        c = subj{i}.c;
        y = subj{i}.y;

        yhat = simulate_subject(theta, t, c, y(1), terms);

        m = min(numel(y), numel(yhat));
        r = [r; (yhat(1:m) - y(1:m))]; %#ok<AGROW>
    end
end


% ==============================================================================
% With Hill(C) terms: theta includes [theta1..thetap, EC50, gamma]
% ==============================================================================
function r = pooled_residual_with_hill(theta, subj, terms)
    p = numel(terms);
    theta_struct = theta(1:p);
    EC50  = theta(p+1);
    gamma = theta(p+2);

    r = [];
    for i = 1:numel(subj)
        t = subj{i}.t;
        c = subj{i}.c;
        y = subj{i}.y;

        yhat = simulate_subject_with_hill(theta_struct, t, c, y(1), terms, EC50, gamma);

        m = min(numel(y), numel(yhat));
        r = [r; (yhat(1:m) - y(1:m))]; %#ok<AGROW>
    end
end


% ==============================================================================
% Multi-equation pooled residuals
% ==============================================================================
function r = pooled_residual_multi(theta, subj, eq_terms)
    r = [];
    for i = 1:numel(subj)
        t = subj{i}.t;
        c = subj{i}.c;
        y = subj{i}.y;

        yhat = simulate_subject_multi(theta, t, c, y(1), eq_terms, 4.0, 2.0);

        m = min(numel(y), numel(yhat));
        r = [r; (yhat(1:m) - y(1:m))]; %#ok<AGROW>
    end
end


function r = pooled_residual_multi_with_hill(theta, subj, eq_terms)
    p = sum(cellfun(@numel, eq_terms));
    theta_struct = theta(1:p);
    EC50  = theta(p+1);
    gamma = theta(p+2);

    r = [];
    for i = 1:numel(subj)
        t = subj{i}.t;
        c = subj{i}.c;
        y = subj{i}.y;

        yhat = simulate_subject_multi(theta_struct, t, c, y(1), eq_terms, EC50, gamma);

        m = min(numel(y), numel(yhat));
        r = [r; (yhat(1:m) - y(1:m))]; %#ok<AGROW>
    end
end


% ==============================================================================
% Standard ODE (EC50/gamma not part of theta)
% ==============================================================================
function yhat = simulate_subject(theta, t, c, R0, terms)
    yhat = simulate_subject_single(theta, t, c, R0, terms, 4.0, 2.0);
end


% ==============================================================================
% ODE with Hill(C) terms: EC50/gamma passed in
% ==============================================================================
function yhat = simulate_subject_with_hill(theta, t, c, R0, terms, EC50, gamma)
    yhat = simulate_subject_single(theta, t, c, R0, terms, EC50, gamma);
end


% ==============================================================================
% Single-equation simulation
% ==============================================================================
function yhat = simulate_subject_single(theta, t, c, R0, terms, EC50, gamma)
    t = t(:); c = c(:);
    [t, ord] = sort(t); c = c(ord);

    [tu, ia] = unique(t, 'stable');
    cu = c(ia);

    if numel(tu) < 2
        yhat = ones(size(t));
        return;
    end

    Cfun = @(tt) interp1(tu, cu, tt, 'linear', 'extrap');
    ode = @(tt, R) rhs_single(tt, R, Cfun, theta, terms, EC50, gamma);
    [tsol, Rsol] = ode45(ode, [tu(1) tu(end)], R0);
    yhat = interp1(tsol, Rsol(:), t, 'linear', 'extrap');
    yhat = yhat(:);
end


% ==============================================================================
% Multi-equation simulation (Eq1=R, Eq2=CpR, Eq3=Ct)
% ==============================================================================
function yhat = simulate_subject_multi(theta, t, c, R0, eq_terms, EC50, gamma)
    t = t(:); c = c(:);
    [t, ord] = sort(t); c = c(ord);

    [tu, ia] = unique(t, 'stable');
    cu = c(ia);

    if numel(tu) < 2
        yhat = ones(size(t));
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
end


% ==============================================================================
% Single-equation RHS
% ==============================================================================
function dR = rhs_single(t, R, Cfun, theta, terms, EC50, gamma)
    ctx.t = t;
    ctx.R = R(1);
    ctx.C = Cfun(t); ctx.C = ctx.C(1);
    ctx.CpR = 0.0;
    ctx.Ct = 0.0;
    dR = sum_terms(theta, terms, ctx, EC50, gamma);
end


% ==============================================================================
% Multi-equation RHS
% ==============================================================================
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
            theta_i = theta(offset + (1:ni));
            dX(iEq) = sum_terms(theta_i, eq_i, ctx, EC50, gamma);
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

            tok = regexp(key, '^([a-z][a-z0-9]*)\*r$', 'tokens', 'once');
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


% ==============================================================================
% 工具：从 JSON decode 得到的 struct 中读取浮点字段
% 支持 candidate struct 内直接有 ec50_hat/gamma_hat 字段
% ==============================================================================
function v = json_float_field(cand, field, default_)
    v = default_;
    if isstruct(cand) && isfield(cand, field)
        val = cand.(field);
        if ~isempty(val)
            if iscell(val), val = val{1}; end
            if isnumeric(val)
                v = max(double(val), eps);
            elseif ischar(val)
                v = max(str2double(val), eps);
            end
        end
    end
end


% ==============================================================================
% 工具：解析候选 terms（支持单方程与多方程）
% ==============================================================================
function [eq_terms, flat_terms, terms_label] = parse_candidate_terms(raw_terms)
    if iscell(raw_terms) && ~isempty(raw_terms)
        is_single_eq = all(cellfun(@is_text_term, raw_terms));
        if is_single_eq
            eq_terms = {to_cellstr_terms(raw_terms)};
            flat_terms = eq_terms{1};
            terms_label = strjoin(flat_terms, ' + ');
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

    eq_terms = {to_cellstr_terms(raw_terms)};
    flat_terms = eq_terms{1};
    terms_label = strjoin(flat_terms, ' + ');
end


function tf = is_text_term(x)
    tf = ischar(x) || (isstring(x) && isscalar(x));
end


% ==============================================================================
% 工具：将 char/string/cell 统一为 row cellstr
% ==============================================================================
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


% ==============================================================================
% 工具：读取候选 rank（缺失时回退到循环序号）
% ==============================================================================
function r = cand_rank(cand, fallback_rank)
    r = fallback_rank;
    if ~(isstruct(cand) && isfield(cand, 'rank'))
        return;
    end

    val = cand.rank;
    if iscell(val) && ~isempty(val)
        val = val{1};
    end
    if isnumeric(val) && isscalar(val) && isfinite(val)
        r = double(val);
        return;
    end
    if ischar(val) || (isstring(val) && isscalar(val))
        v = str2double(string(val));
        if ~isnan(v)
            r = double(v);
        end
    end
end
