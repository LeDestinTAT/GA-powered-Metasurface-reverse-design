%% 参数设置
N_pool   = 30000;   % 候选池大小（不跑 COMSOL）
N_target = 5000;    % 最终训练集数量

fill_min = 0.2;
fill_max = 0.8;

ham_big  = 8;   % 全局差异阈值
ham_small = 3;  % 局部突变阈值

ratio_big = 0.7;   % 70% 全局差异
ratio_small = 0.3;

rng(1); % 固定随机种子，方便复现

%% Step 1: 生成候选池
patterns = false(11,11,N_pool);
fills = zeros(N_pool,1);
bounds = zeros(N_pool,1);

cnt = 0;
while cnt < N_pool
    p = rand(11,11) > 0.5;
    f = mean(p(:));
    if f > fill_min && f < fill_max
        cnt = cnt + 1;
        patterns(:,:,cnt) = p;
        fills(cnt) = f;
        bounds(cnt) = ...
            sum(sum(abs(diff(p,1,1)))) + ...
            sum(sum(abs(diff(p,1,2))));
    end
end

%% Step 2: 分桶（填充率 × 边界复杂度）
n_fill_bins = 10;
n_bound_bins = 10;

fill_edges  = linspace(fill_min, fill_max, n_fill_bins+1);
bound_edges = linspace(min(bounds), max(bounds), n_bound_bins+1);

bucket_id = zeros(N_pool,1);
for i = 1:N_pool
    bf = find(fills(i) >= fill_edges(1:end-1) & fills(i) < fill_edges(2:end),1);
    bb = find(bounds(i) >= bound_edges(1:end-1) & bounds(i) < bound_edges(2:end),1);
    if isempty(bf), bf = n_fill_bins; end
    if isempty(bb), bb = n_bound_bins; end
    bucket_id(i) = (bf-1)*n_bound_bins + bb;
end

unique_buckets = unique(bucket_id);
n_bucket = numel(unique_buckets);

%% Step 3: 桶内双通道筛选
selected = false(11,11,0);
selected_vec = [];

for b = unique_buckets'
    idx = find(bucket_id == b);
    if isempty(idx), continue; end

    n_b = round(N_target / n_bucket);
    n_big = round(n_b * ratio_big);
    n_small = n_b - n_big;

    % --- 大差异样本 ---
    perm = idx(randperm(numel(idx)));
    for i = perm'
        if size(selected,3) >= N_target, break; end
        v = patterns(:,:,i); v = v(:)';
        if isempty(selected_vec)
            selected(:,:,end+1) = patterns(:,:,i);
            selected_vec(end+1,:) = v;
        else
            d = min(sum(xor(selected_vec, v),2));
            if d >= ham_big
                selected(:,:,end+1) = patterns(:,:,i);
                selected_vec(end+1,:) = v;
            end
        end
        if size(selected,3) >= sum(bucket_id <= b) * n_big, break; end
    end

    % --- 小差异但关键突变 ---
    perm = idx(randperm(numel(idx)));
    for i = perm'
        if size(selected,3) >= N_target, break; end
        v = patterns(:,:,i); v = v(:)';
        if isempty(selected_vec)
            continue;
        end
        d = min(sum(xor(selected_vec, v),2));

        % 中心像素突变永远保留
        center_changed = any(any(xor(patterns(5:7,5:7,i), ...
                                         selected(5:7,5:7,end))));
        if (d <= ham_small) || center_changed
            selected(:,:,end+1) = patterns(:,:,i);
            selected_vec(end+1,:) = v;
        end
        if size(selected,3) >= sum(bucket_id <= b) * n_b, break; end
    end
end

%% Step 4: 若不足，随机补齐（不再过滤）
if size(selected,3) < N_target
    remain = N_target - size(selected,3);
    rest_idx = setdiff(1:N_pool, 1:size(selected,3));
    perm = rest_idx(randperm(numel(rest_idx), remain));
    for i = perm
        selected(:,:,end+1) = patterns(:,:,i);
    end
end

%% Step 5: 保存结果
selected = selected(:,:,1:N_target);
save('training_patterns_11x11.mat', 'selected');

fprintf('已生成 %d 个训练样本\n', size(selected,3));
