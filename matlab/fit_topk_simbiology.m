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
%   rank, terms, converged, logLik, AIC, BIC, RMSE, SSE, message, theta1, ..., thetaK, EC50, gamma,
%   n_starts, n_success, best_start_idx, stability_cv

    if nargin < 3
        out_csv = 'artifacts/nlme/simbiology_results.csv';
    end

    T = readtable(data_csv);
    T.Properties.VariableNames = {'sid','time','C_obs','R_obs'};
    S = jsondecode(fileread(cand_json));
    nlme_mode = read_nlme_mode(S);
    nlme_multistart_on_fail = read_nlme_multistart_on_fail(S);
    fit_hints = read_fit_hints(S);
    [direct_sigemax_case, dh_ec50_lb, dh_ec50_ub, dh_gamma_lb, dh_gamma_ub] = read_direct_hill_bounds(fit_hints);
    hill_gamma_grid = read_hill_gamma_grid(fit_hints, dh_gamma_lb, dh_gamma_ub);
    struct_theta_abs_bound = read_struct_theta_abs_bound(fit_hints, 30.0);
    if strcmpi(nlme_mode, "confirm")
        n_starts_tmdd = 8;
        n_starts_non_tmdd = 4;
        fallback_max_iterations = 500;
        max_function_evaluations = 1200;
        fallback_function_tolerance = 1e-9;
        fallback_step_tolerance = 1e-9;
    else
        n_starts_tmdd = 4;
        n_starts_non_tmdd = 3;
        fallback_max_iterations = 180;
        max_function_evaluations = 350;
        fallback_function_tolerance = 1e-6;
        fallback_step_tolerance = 1e-6;
    end
    single_max_iterations = 180;
    single_function_tolerance = 1e-6;
    single_step_tolerance = 1e-6;

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
    if direct_sigemax_case
        struct_theta_abs_bound = max(struct_theta_abs_bound, estimate_response_scaled_theta_bound(T.R_obs));
    end

    cand_items = normalize_candidate_items(S);
    nCand = numel(cand_items);
    if nCand < 1
        error('fit_topk_simbiology:NoCandidates', 'No valid candidates found in cand_json.');
    end

    % ---- 第一遍：确定所有候选的最大维数 ----
    % 包括结构项 theta，以及可选的 EC50/gamma
    max_k = 0;
    any_hill_model = false;
    for j = 1:nCand
        cand = cand_items{j};
        raw_terms = candidate_terms_field(cand);
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
    %          theta1..thetaK, [EC50, gamma] (only if any_hill_model), multistart diagnostics
    nVar = 9 + max_k + 2*double(any_hill_model) + 4;
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
        varNames{10+max_k} = 'EC50';
        varNames{11+max_k} = 'gamma';
    end
    varNames{nVar-3} = 'n_starts';
    varNames{nVar-2} = 'n_success';
    varNames{nVar-1} = 'best_start_idx';
    varNames{nVar}   = 'stability_cv';

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
        cand = cand_items{j};
        raw_terms = candidate_terms_field(cand);
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

                ec50_lb = 0.5;
                ec50_ub = 20.0;
                gamma_lb = 0.5;
                gamma_ub = 6.0;
                if direct_sigemax_case
                    ec50_lb = dh_ec50_lb;
                    ec50_ub = dh_ec50_ub;
                    gamma_lb = dh_gamma_lb;
                    gamma_ub = dh_gamma_ub;
                end
                theta0(p+1) = min(max(theta0(p+1), ec50_lb), ec50_ub);
                theta0(p+2) = min(max(theta0(p+2), gamma_lb), gamma_ub);

                struct_bounds = struct_theta_abs_bound * ones(p, 1);
                if direct_sigemax_case
                    struct_bounds = direct_struct_bounds(flat_terms, struct_theta_abs_bound);
                end
                lb = [-struct_bounds; ec50_lb; gamma_lb];
                ub = [ struct_bounds; ec50_ub; gamma_ub];

                if is_multi_eq
                    obj = @(th) pooled_residual_multi_with_hill(th, subj, eq_terms, hill_gamma_grid);
                else
                    obj = @(th) pooled_residual_with_hill(th, subj, flat_terms, hill_gamma_grid);
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
                struct_bounds = struct_theta_abs_bound * ones(p, 1);
                if direct_sigemax_case
                    struct_bounds = direct_struct_bounds(flat_terms, struct_theta_abs_bound);
                end
                lb = -struct_bounds;
                ub =  struct_bounds;
                if is_multi_eq
                    obj = @(th) pooled_residual_multi(th, subj, eq_terms);
                else
                    obj = @(th) pooled_residual(th, subj, flat_terms);
                end
            end

            opts_multistart = optimoptions('lsqnonlin', ...
                'Display', 'off', ...
                'MaxIterations', fallback_max_iterations, ...
                'MaxFunctionEvaluations', max_function_evaluations, ...
                'FunctionTolerance', fallback_function_tolerance, ...
                'StepTolerance', fallback_step_tolerance);
            use_tmdd_multistart = candidate_has_tmdd_terms(flat_terms);
            n_start_plan = n_starts_non_tmdd;
            if use_tmdd_multistart
                n_start_plan = n_starts_tmdd;
            end
            start_matrix = [];
            if has_EC50_gamma
                n_start_plan = max(n_start_plan, numel(hill_gamma_grid));
                start_matrix = build_hill_start_matrix(theta0, lb, ub, n_start_plan, direct_sigemax_case, hill_gamma_grid);
            end
            [theta_hat, resid, exitflag, n_starts, n_success, best_start_idx, stability_cv] = ...
                fit_with_optional_multistart(obj, theta0, lb, ub, opts_multistart, n_start_plan, start_matrix);
            if has_EC50_gamma
                theta_hat(p+2) = snap_to_grid(theta_hat(p+2), hill_gamma_grid);
            end

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
            if exitflag > 0
                rowCell{9} = sprintf("converged_multistart(n_starts=%d,n_success=%d,best_start=%g,stability_cv=%g)", ...
                    n_starts, n_success, best_start_idx, stability_cv);
            else
                rowCell{9} = sprintf("nonconverged_multistart(exitflag=%d,n_starts=%d,n_success=%d,best_start=%g,stability_cv=%g)", ...
                    exitflag, n_starts, n_success, best_start_idx, stability_cv);
            end
            for v = 10:nVar
                rowCell{v} = NaN;
            end
            for k = 1:p
                rowCell{9+k} = theta_hat(k);
            end
            if has_EC50_gamma && any_hill_model
                rowCell{10+max_k} = theta_hat(p+1);  % EC50
                rowCell{11+max_k} = theta_hat(p+2);  % gamma
            end
            rowCell{nVar-3} = n_starts;
            rowCell{nVar-2} = n_success;
            rowCell{nVar-1} = best_start_idx;
            rowCell{nVar} = stability_cv;
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

        yhat = simulate_subject(theta, t, c, estimate_initial_state(y), terms);

        m = min(numel(y), numel(yhat));
        r = [r; (yhat(1:m) - y(1:m))]; %#ok<AGROW>
    end
end


% ==============================================================================
% With Hill(C) terms: theta includes [theta1..thetap, EC50, gamma]
% ==============================================================================
function r = pooled_residual_with_hill(theta, subj, terms, gamma_grid)
    p = numel(terms);
    theta_struct = theta(1:p);
    EC50  = theta(p+1);
    gamma = snap_to_grid(theta(p+2), gamma_grid);

    r = [];
    for i = 1:numel(subj)
        t = subj{i}.t;
        c = subj{i}.c;
        y = subj{i}.y;

        yhat = simulate_subject_with_hill(theta_struct, t, c, estimate_initial_state(y), terms, EC50, gamma);

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

        yhat = simulate_subject_multi(theta, t, c, estimate_initial_state(y), eq_terms, 4.0, 2.0);

        m = min(numel(y), numel(yhat));
        r = [r; (yhat(1:m) - y(1:m))]; %#ok<AGROW>
    end
end


function r = pooled_residual_multi_with_hill(theta, subj, eq_terms, gamma_grid)
    p = sum(cellfun(@numel, eq_terms));
    theta_struct = theta(1:p);
    EC50  = theta(p+1);
    gamma = snap_to_grid(theta(p+2), gamma_grid);

    r = [];
    for i = 1:numel(subj)
        t = subj{i}.t;
        c = subj{i}.c;
        y = subj{i}.y;

        yhat = simulate_subject_multi(theta_struct, t, c, estimate_initial_state(y), eq_terms, EC50, gamma);

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
% 工具：标准化 JSON 候选容器（兼容 struct array / cell）
% ==============================================================================
function items = normalize_candidate_items(S)
    items = {};
    if ~(isstruct(S) && isfield(S, 'candidates')) || isempty(S.candidates)
        return;
    end
    c = S.candidates;
    if iscell(c)
        items = c(:);
    elseif isstruct(c)
        items = arrayfun(@(x) {x}, c(:));
    end
end


function raw_terms = candidate_terms_field(cand)
    if iscell(cand) && numel(cand) == 1
        cand = cand{1};
    end
    if ~(isstruct(cand) && isfield(cand, 'terms'))
        error('fit_topk_simbiology:InvalidCandidate', 'Each candidate must be a struct with field ''terms''.');
    end
    raw_terms = cand.terms;
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
% TMDD-aware fitting helpers
% ==============================================================================
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


function [theta_best, resid_best, exitflag_best, n_starts, n_success, best_start_idx, stability_cv] = ...
    fit_with_optional_multistart(obj, theta0, lb, ub, opts, n_starts, start_matrix)

    theta0 = theta0(:);
    lb = lb(:);
    ub = ub(:);
    n_param = numel(theta0);

    if nargin < 6 || isempty(n_starts) || n_starts < 1
        n_starts = 1;
    end
    if nargin < 7
        start_matrix = [];
    end
    n_seed = 0;
    if ~isempty(start_matrix) && isnumeric(start_matrix)
        if size(start_matrix, 1) == n_param
            n_seed = size(start_matrix, 2);
            n_starts = max(n_starts, n_seed);
        else
            start_matrix = [];
        end
    else
        start_matrix = [];
    end

    success_sse = [];
    theta_best = theta0;
    resid_best = [];
    exitflag_best = -1;
    best_sse = inf;
    best_start_idx = NaN;
    n_success = 0;

    for iStart = 1:n_starts
        if iStart <= n_seed
            start_i = start_matrix(:, iStart);
        else
            start_i = theta0;
            span = ub - lb;
            span(~isfinite(span) | span <= 0) = 1.0;
            jitter = 0.20 .* span .* randn(n_param, 1);
            start_i = theta0 + jitter;
        end
        start_i = min(max(start_i, lb), ub);

        try
            [theta_i, ~, resid_i, exitflag_i] = lsqnonlin(obj, start_i, lb, ub, opts); %#ok<ASGLU>
            if exitflag_i > 0 && all(isfinite(resid_i))
                sse_i = sum(resid_i(:).^2);
                if isfinite(sse_i)
                    n_success = n_success + 1;
                    success_sse(end+1, 1) = sse_i; %#ok<AGROW>
                    if sse_i < best_sse
                        best_sse = sse_i;
                        theta_best = theta_i;
                        resid_best = resid_i;
                        exitflag_best = exitflag_i;
                        best_start_idx = iStart;
                    end
                end
            end
        catch
            % skip failed start
        end
    end

    if n_success == 0
        [theta_best, ~, resid_best, exitflag_best] = lsqnonlin(obj, theta0, lb, ub, opts); %#ok<ASGLU>
        n_success = double(exitflag_best > 0);
        best_start_idx = 1;
    end

    stability_cv = NaN;
    if numel(success_sse) >= 2
        mu = mean(success_sse);
        if abs(mu) > eps
            stability_cv = std(success_sse) / abs(mu);
        end
    end
end


function start_matrix = build_hill_start_matrix(theta0, lb, ub, n_starts, narrow_bounds, gamma_grid)
    theta0 = theta0(:);
    lb = lb(:);
    ub = ub(:);
    n_param = numel(theta0);
    if n_param < 3
        start_matrix = [];
        return;
    end
    p = n_param - 2;
    ec50_lb = lb(p+1);
    ec50_ub = ub(p+1);
    gamma_lb = lb(p+2);
    gamma_ub = ub(p+2);
    ec50_mid = 0.5 * (ec50_lb + ec50_ub);
    gamma_mid = 0.5 * (gamma_lb + gamma_ub);
    ec50_vals = unique([theta0(p+1), ec50_lb, ec50_mid, ec50_ub]);
    gamma_vals = unique([theta0(p+2), gamma_lb, gamma_mid, gamma_ub]);
    if nargin >= 6 && ~isempty(gamma_grid)
        gamma_vals = unique(gamma_grid(:)');
        gamma_vals = gamma_vals(gamma_vals >= gamma_lb & gamma_vals <= gamma_ub);
        if isempty(gamma_vals)
            gamma_vals = unique([theta0(p+2), gamma_mid]);
        end
    end
    if narrow_bounds
        ec50_vals = unique([theta0(p+1), ec50_mid, ec50_lb + 0.75*(ec50_ub-ec50_lb)]);
        if nargin < 6 || isempty(gamma_grid)
            gamma_vals = unique([theta0(p+2), gamma_mid, gamma_ub]);
        end
    end
    starts = zeros(n_param, max(1, n_starts));
    starts(:, 1) = theta0;
    idx = 2;
    for ie = 1:numel(ec50_vals)
        for ig = 1:numel(gamma_vals)
            if idx > n_starts
                break;
            end
            s = theta0;
            s(p+1) = ec50_vals(ie);
            s(p+2) = gamma_vals(ig);
            starts(:, idx) = s;
            idx = idx + 1;
        end
        if idx > n_starts
            break;
        end
    end
    for k = idx:n_starts
        span = ub - lb;
        span(~isfinite(span) | span <= 0) = 1.0;
        s = theta0 + 0.20 .* span .* randn(n_param, 1);
        starts(:, k) = min(max(s, lb), ub);
    end
    start_matrix = starts;
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


function mode = read_nlme_mode(S)
    mode = "screen";
    if ~(isstruct(S) && isfield(S, 'nlme_mode'))
        return;
    end
    raw = S.nlme_mode;
    if iscell(raw) && ~isempty(raw)
        raw = raw{1};
    end
    if ischar(raw) || (isstring(raw) && isscalar(raw))
        val = strtrim(string(raw));
        if strcmpi(val, "confirm")
            mode = "confirm";
        end
    end
end


function tf = read_nlme_multistart_on_fail(S)
    tf = true;
    if ~(isstruct(S) && isfield(S, 'nlme_multistart_on_fail'))
        return;
    end

    raw = S.nlme_multistart_on_fail;
    if iscell(raw) && ~isempty(raw)
        raw = raw{1};
    end

    if islogical(raw) && isscalar(raw)
        tf = raw;
        return;
    end
    if isnumeric(raw) && isscalar(raw) && isfinite(raw)
        tf = (double(raw) ~= 0);
        return;
    end
    if ischar(raw) || (isstring(raw) && isscalar(raw))
        val = lower(strtrim(string(raw)));
        if strcmp(val, "true") || strcmp(val, "1")
            tf = true;
        elseif strcmp(val, "false") || strcmp(val, "0")
            tf = false;
        end
    end
end


function H = read_fit_hints(S)
    H = struct();
    if ~(isstruct(S) && isfield(S, 'fit_hints'))
        return;
    end
    raw = S.fit_hints;
    if iscell(raw) && ~isempty(raw)
        raw = raw{1};
    end
    if isstruct(raw)
        H = raw;
    end
end


function [is_direct_case, ec50_lb, ec50_ub, gamma_lb, gamma_ub] = read_direct_hill_bounds(H)
    is_direct_case = false;
    ec50_lb = 0.5;
    ec50_ub = 8.0;
    gamma_lb = 0.5;
    gamma_ub = 3.0;
    if ~isstruct(H)
        return;
    end
    if isfield(H, 'direct_sigemax_case')
        is_direct_case = parse_bool_like(H.direct_sigemax_case, false);
    end
    if isfield(H, 'direct_hill_bounds')
        B = H.direct_hill_bounds;
        if iscell(B) && ~isempty(B)
            B = B{1};
        end
        if isstruct(B)
            ec50_lb = parse_float_like(get_struct_field(B, 'ec50_lb'), ec50_lb);
            ec50_ub = parse_float_like(get_struct_field(B, 'ec50_ub'), ec50_ub);
            gamma_lb = parse_float_like(get_struct_field(B, 'gamma_lb'), gamma_lb);
            gamma_ub = parse_float_like(get_struct_field(B, 'gamma_ub'), gamma_ub);
        end
    end
    ec50_lb = max(ec50_lb, eps);
    ec50_ub = max(ec50_ub, ec50_lb + 1e-6);
    gamma_lb = max(gamma_lb, eps);
    gamma_ub = max(gamma_ub, gamma_lb + 1e-6);
end


function v = read_struct_theta_abs_bound(H, default_v)
    v = default_v;
    if ~isstruct(H)
        return;
    end
    if isfield(H, 'struct_theta_abs_bound')
        v = parse_float_like(H.struct_theta_abs_bound, default_v);
    end
    v = min(max(v, 30.0), 5000.0);
end


function grid = read_hill_gamma_grid(H, gamma_lb, gamma_ub)
    grid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
    if isstruct(H) && isfield(H, 'hill_gamma_grid')
        raw = H.hill_gamma_grid;
        if iscell(raw)
            raw = cellfun(@double, raw);
        end
        if isnumeric(raw)
            vals = double(raw(:)');
            vals = vals(isfinite(vals));
            if ~isempty(vals)
                grid = vals;
            end
        end
    end
    grid = unique(grid);
    grid = grid(grid >= gamma_lb & grid <= gamma_ub);
    if isempty(grid)
        grid = unique([gamma_lb, 0.5 * (gamma_lb + gamma_ub), gamma_ub]);
    end
end


function v = snap_to_grid(x, grid)
    if isempty(grid)
        v = x;
        return;
    end
    vals = double(grid(:));
    [~, idx] = min(abs(vals - double(x)));
    v = vals(idx);
end


function v = estimate_response_scaled_theta_bound(r_values)
    vals = double(r_values(:));
    vals = vals(isfinite(vals));
    if isempty(vals)
        v = 30.0;
        return;
    end
    max_abs = max(abs(vals));
    dyn = max(vals) - min(vals);
    v = min(max([30.0, 6.0 * max_abs, 12.0 * dyn]), 5000.0);
end


function bounds = direct_struct_bounds(terms, large_bound)
    bounds = large_bound * ones(numel(terms), 1);
    for k = 1:numel(terms)
        key = lower(strrep(strtrim(char(string(terms{k}))), ' ', ''));
        if strcmp(key, '1') || strcmp(key, 'emax(c)') || strcmp(key, 'hill(c)')
            b = large_bound;
        elseif strcmp(key, 'r') || strcmp(key, 'c')
            b = min(large_bound, 50.0);
        elseif strcmp(key, 'c^2')
            b = min(large_bound, 10.0);
        elseif contains(key, '*r')
            b = min(large_bound, 10.0);
        else
            b = min(large_bound, 50.0);
        end
        bounds(k) = max(b, 1e-6);
    end
end


function v = get_struct_field(S, name)
    v = [];
    if isstruct(S) && isfield(S, name)
        v = S.(name);
    end
end


function v = parse_float_like(x, default_v)
    v = default_v;
    if isempty(x)
        return;
    end
    if iscell(x) && ~isempty(x)
        x = x{1};
    end
    if isnumeric(x) && isscalar(x) && isfinite(x)
        v = double(x);
        return;
    end
    if ischar(x) || (isstring(x) && isscalar(x))
        y = str2double(string(x));
        if ~isnan(y)
            v = double(y);
        end
    end
end


function tf = parse_bool_like(x, default_tf)
    tf = default_tf;
    if isempty(x)
        return;
    end
    if iscell(x) && ~isempty(x)
        x = x{1};
    end
    if islogical(x) && isscalar(x)
        tf = x;
        return;
    end
    if isnumeric(x) && isscalar(x) && isfinite(x)
        tf = (double(x) ~= 0);
        return;
    end
    if ischar(x) || (isstring(x) && isscalar(x))
        val = lower(strtrim(string(x)));
        if strcmp(val, "true") || strcmp(val, "1")
            tf = true;
        elseif strcmp(val, "false") || strcmp(val, "0")
            tf = false;
        end
    end
end


function r0 = estimate_initial_state(y)
    y = y(:); n = min(3, numel(y));
    if n==0, r0 = 1.0; return; end
    r0 = median(y(1:n));
    if ~isfinite(r0), r0 = y(1); end
    if ~isfinite(r0), r0 = 1.0; end
end
