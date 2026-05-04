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
    max_theta_with_hill = 0;
    any_hill_model = false;
    for j = 1:nCand
        terms = S.candidates(j).terms;
        p = numel(terms);
        has_hill = any(strcmp(terms, 'Hill(C)')) || any(strcmp(terms, 'Hill(C)*R'));
        if has_hill
            any_hill_model = true;
            max_theta_with_hill = max(max_theta_with_hill, p);
            max_k = max(max_k, p + 2);  % +2 for EC50/gamma
        else
            max_k = max(max_k, p);
        end
    end

    % ---- 预分配 table（统一所有候选的列）----
    % Columns: rank, terms, converged, logLik, AIC, BIC, RMSE, SSE, message,
    %          theta1..thetaK, [EC50, gamma] (only if any_hill_model)
    if any_hill_model
        nVar = 9 + max_k + 2;  % 9 fixed + max_k thetas + EC50 + gamma
    else
        nVar = 9 + max_k;      % 9 fixed + max_k thetas (no EC50/gamma)
    end
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
        terms = cand.terms;   % already a cell array from JSON decode
        p = numel(terms);

        has_hill = any(strcmp(terms, 'Hill(C)')) || any(strcmp(terms, 'Hill(C)*R'));

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

                obj = @(th) pooled_residual_with_hill(th, subj, terms);
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
                obj = @(th) pooled_residual(th, subj, terms);
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
            rowCell{1} = double(cand.rank);
            rowCell{2} = strjoin(terms, ' + ');
            rowCell{3} = (exitflag > 0);
            rowCell{4} = logLik;
            rowCell{5} = AIC;
            rowCell{6} = BIC;
            rowCell{7} = rmse;
            rowCell{8} = sse;
            rowCell{9} = '';

            for k = 1:max_k
                rowCell{9+k} = NaN;
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
            rowCell{1} = double(cand.rank);
            rowCell{2} = strjoin(terms, ' + ');
            rowCell{3} = false;
            rowCell{4} = NaN;
            rowCell{5} = NaN;
            rowCell{6} = NaN;
            rowCell{7} = NaN;
            rowCell{8} = NaN;
            rowCell{9} = errMsg;
            for k = 1:max_k
                rowCell{9+k} = NaN;
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
% Standard ODE (EC50/gamma not part of theta)
% ==============================================================================
function yhat = simulate_subject(theta, t, c, R0, terms)
    t = t(:); c = c(:);
    [t, ord] = sort(t); c = c(ord);

    [tu, ia] = unique(t, 'stable');
    cu = c(ia);

    if numel(tu) < 2
        yhat = ones(size(t));
        return;
    end

    Cfun = @(tt) interp1(tu, cu, tt, 'linear', 'extrap');
    ode = @(tt, R) rhs(tt, R, Cfun, theta, terms);
    [tsol, Rsol] = ode45(ode, [tu(1) tu(end)], R0);
    yhat = interp1(tsol, Rsol(:), t, 'linear', 'extrap');
    yhat = yhat(:);
end


% ==============================================================================
% ODE with Hill(C) terms: EC50/gamma passed in
% ==============================================================================
function yhat = simulate_subject_with_hill(theta, t, c, R0, terms, EC50, gamma)
    t = t(:); c = c(:);
    [t, ord] = sort(t); c = c(ord);

    [tu, ia] = unique(t, 'stable');
    cu = c(ia);

    if numel(tu) < 2
        yhat = ones(size(t));
        return;
    end

    Cfun = @(tt) interp1(tu, cu, tt, 'linear', 'extrap');
    ode = @(tt, R) rhs_with_hill(tt, R, Cfun, theta, terms, EC50, gamma);
    [tsol, Rsol] = ode45(ode, [tu(1) tu(end)], R0);
    yhat = interp1(tsol, Rsol(:), t, 'linear', 'extrap');
    yhat = yhat(:);
end


% ==============================================================================
% Standard RHS (EC50/gamma = 4.0/2.0 fixed)
% ==============================================================================
function dR = rhs(t, R, Cfun, theta, terms)
    R = R(1);
    C = Cfun(t);
    C = max(C(1), 1e-10);

    EC50 = 4.0;
    gamma = 2.0;

    HillC = (C^gamma)/(EC50^gamma + C^gamma);

    lib = containers.Map('KeyType','char','ValueType','double');
    lib('1') = 1.0;
    lib('R') = R;
    lib('C') = C;
    lib('C^2') = C^2;
    lib('Emax(C)') = C/(EC50 + C);
    lib('Hill(C)') = HillC;
    lib('C*R') = C*R;
    lib('Emax(C)*R') = (C/(EC50 + C))*R;
    lib('Hill(C)*R') = HillC*R;

    dR = 0.0;
    for k = 1:numel(terms)
        key = char(terms{k});
        if isKey(lib, key)
            dR = dR + theta(k)*lib(key);
        end
    end
end


% ==============================================================================
% Hill(C) RHS: EC50/gamma passed as parameters
% ==============================================================================
function dR = rhs_with_hill(t, R, Cfun, theta, terms, EC50, gamma)
    R = R(1);
    C = Cfun(t);
    C = max(C(1), 1e-10);

    HillC = (C^gamma)/(EC50^gamma + C^gamma);

    lib = containers.Map('KeyType','char','ValueType','double');
    lib('1') = 1.0;
    lib('R') = R;
    lib('C') = C;
    lib('C^2') = C^2;
    lib('Emax(C)') = C/(EC50 + C);
    lib('Hill(C)') = HillC;
    lib('C*R') = C*R;
    lib('Emax(C)*R') = (C/(EC50 + C))*R;
    lib('Hill(C)*R') = HillC*R;

    dR = 0.0;
    for k = 1:numel(terms)
        key = char(terms{k});
        if isKey(lib, key)
            dR = dR + theta(k)*lib(key);
        end
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