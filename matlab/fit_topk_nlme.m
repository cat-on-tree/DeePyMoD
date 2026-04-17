function results = fit_topk_nlme(data_csv, cand_json, out_csv)
% 批量拟合 Top-K 候选结构 (nlmefit, cell-input 版本，避免维度错配)
%
% 输入:
%   data_csv  : pkpd_long.csv，列: sid,time,C_obs,R_obs
%   cand_json : topk_candidates.json
%   out_csv   : 输出结果csv
%
% 输出:
%   results(table): rank,terms,converged,logLik,AIC,BIC,MSE,message

    if nargin < 3
        out_csv = 'artifacts/nlme/nlme_results.csv';
    end

    T = readtable(data_csv);
    S = jsondecode(fileread(cand_json));

    sid = T.sid;
    t   = T.time;
    C   = T.C_obs;
    R   = T.R_obs;

    [sid_u, ~, sid_idx] = unique(sid);
    nSub = numel(sid_u);

    % 按受试者组织成 cell（nlmefit 推荐格式之一）
    Xi = cell(nSub, 1);   % 每个受试者的 X: [time, C_obs]
    Yi = cell(nSub, 1);   % 每个受试者的 Y: R_obs
    for i = 1:nSub
        idx = (sid_idx == i);
        ti = t(idx);
        ci = C(idx);
        yi = R(idx);

        [ti, ord] = sort(ti);
        ci = ci(ord);
        yi = yi(ord);

        Xi{i} = [ti(:), ci(:)];
        Yi{i} = yi(:);
    end

    nCand = numel(S.candidates);
    rows = struct([]);

    for j = 1:nCand
        cand = S.candidates(j);

        % JSON 读出来可能是 string array / cellstr，统一成 cellstr
        terms = cellstr(cand.terms);
        p = numel(terms);
        beta0 = zeros(1, p);

        % 模型函数：phi(1xp) + x_i(nx2) -> yhat_i(nx1)
        modelfun = @(phi, x_i) subject_pred(phi, x_i, terms);

        try
            opts = statset('nlmefit');
            opts.MaxIter = 300;
            opts.Display = 'off';

            % cell 输入形式（关键：避免 stacked+group 带来的尺寸错配）
            [beta, PSI, stats, b] = nlmefit( ...
                Xi, Yi, [], modelfun, beta0, ...
                'REParamsSelect', true(1,p), ...
                'CovPattern', diag(true(1,p)), ...
                'ErrorModel', 'constant', ...
                'Options', opts ...
            );

            % 拟合优度统计
            n = 0;
            for i = 1:nSub
                n = n + numel(Yi{i});
            end

            % 参数个数粗略计数：固定效应 + PSI上三角
            k = numel(beta) + nnz(triu(true(size(PSI))));
            ll = stats.logl;
            aic = -2 * ll + 2 * k;
            bic = -2 * ll + log(n) * k;

            yhat = fitted_from_b(beta, b, Xi, terms);
            rss = 0.0;
            for i = 1:nSub
                yi = Yi{i};
                ypi = yhat{i};
                m = min(numel(yi), numel(ypi)); % 防守式处理
                rss = rss + sum((yi(1:m) - ypi(1:m)).^2);
            end
            mse = rss / n;

            rows(end+1).rank = cand.rank; %#ok<AGROW>
            rows(end).terms = strjoin(terms, ' + ');
            rows(end).converged = true;
            rows(end).logLik = ll;
            rows(end).AIC = aic;
            rows(end).BIC = bic;
            rows(end).MSE = mse;
            rows(end).message = "";

        catch ME
            rows(end+1).rank = cand.rank; %#ok<AGROW>
            rows(end).terms = strjoin(terms, ' + ');
            rows(end).converged = false;
            rows(end).logLik = NaN;
            rows(end).AIC = NaN;
            rows(end).BIC = NaN;
            rows(end).MSE = NaN;
            rows(end).message = string(ME.identifier + " | " + ME.message);
        end
    end

    results = struct2table(rows);

    % 先收敛，再按 BIC 升序
    if ~isempty(results)
        results = sortrows(results, {'converged','BIC'}, {'descend','ascend'});
    end

    out_dir = fileparts(out_csv);
    if ~isempty(out_dir) && ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    writetable(results, out_csv);
end


function yhat_i = subject_pred(phi, x_i, terms)
% x_i: [t, C], phi: 1xp 或 px1
% 输出 yhat_i: nx1
    phi = phi(:);  % 强制列向量

    t = x_i(:,1); t = t(:);
    C = x_i(:,2); C = C(:);

    [t, ord] = sort(t);
    C = C(ord);

    n = numel(t);
    if n < 2
        yhat_i = nan(n,1);
        return;
    end

    % 去重以防插值报错（同一时刻重复点）
    [tu, ia] = unique(t, 'stable');
    Cu = C(ia);

    if numel(tu) < 2
        yhat_i = repmat(1.0, n, 1);
        return;
    end

    Cfun = @(tt) interp1(tu, Cu, tt, 'linear', 'extrap');

    R0 = 1.0; % 简化初值（后续可改为个体首观测）
    ode = @(tt, R) rhs(tt, R, Cfun, phi, terms);

    [tsol, Rsol] = ode45(ode, tu, R0);
    Rsol = Rsol(:);

    % 映射回原 t（含重复点）
    yhat_i = interp1(tsol, Rsol, t, 'linear', 'extrap');
    yhat_i = yhat_i(:);
end


function dR = rhs(t, R, Cfun, phi, terms)
    R = R(1);           % 标量
    C = Cfun(t);
    C = C(1);
    C = max(C, 1e-10);

    % 这里先固定形状参数；后续可扩展成可估计
    EC50 = 4.0;
    gamma = 2.0;

    EmaxC = C / (EC50 + C);
    HillC = (C^gamma) / (EC50^gamma + C^gamma);

    lib = containers.Map('KeyType','char','ValueType','double');
    lib('1') = 1.0;
    lib('R') = R;
    lib('C') = C;
    lib('C^2') = C^2;
    lib('Emax(C)') = EmaxC;
    lib('Hill(C)') = HillC;
    lib('C*R') = C * R;
    lib('Emax(C)*R') = EmaxC * R;
    lib('Hill(C)*R') = HillC * R;

    val = 0.0;
    for k = 1:numel(terms)
        term = char(terms{k});
        if ~isKey(lib, term)
            error('Unknown term in candidate: %s', term);
        end
        val = val + phi(k) * lib(term);
    end

    dR = val;
end


function yhat = fitted_from_b(beta, b, Xi, terms)
% 用个体参数 phi_i = beta + b_i 回代预测
    beta = beta(:);
    nSub = numel(Xi);
    yhat = cell(nSub,1);

    for i = 1:nSub
        bi = b(i,:); bi = bi(:);
        phi_i = beta + bi;
        yhat{i} = subject_pred(phi_i, Xi{i}, terms);
    end
end