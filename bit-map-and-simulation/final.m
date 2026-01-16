model = mphload('single.mph');
geom = model.geom('geom1');



% 删除柱体特征
geom.feature.remove('cyl1');period = 2.8e-6;               % 器件边长 2.8 μm
top_thickness = 100e-9;        % 顶层厚度 100 nm
grid_size = 11;                % 11x11 网格（可根据需要调整）
pixel_spacing = period / grid_size;  % 中心间距 = period / grid_size
cylinder_radius = pixel_spacing / 2; % 半径 = 间距/2 → 相邻柱体精确相切

% ==================== 二值矩阵（'1' = 生成柱体，'0' = 无） ====================
% 示例：中心有柱体，四周部分填充（你可以替换为你的设计）
binary_matrix = [
    0 0 0 0 0 1 1 1 0 0 0;
    0 0 0 0 1 1 1 1 1 0 0;
    0 0 0 1 1 1 1 1 1 1 0;
    0 0 1 1 1 1 1 1 1 1 0;
    0 1 1 1 1 1 1 1 1 1 0;
    1 1 1 1 1 1 1 1 1 1 1;
    1 1 1 1 1 1 1 1 1 1 0;
    0 1 1 1 1 1 1 1 1 0 0;
    0 0 1 1 1 1 1 1 0 0 0;
    0 0 0 1 1 1 1 0 0 0 0;
    0 0 0 0 1 1 0 0 0 0 0];
% ==================== 计算柱体中心坐标（中心对齐） ====================
% 中心位于 (period/2, period/2)
x_starts = period/2 - (grid_size-1)*pixel_spacing/2 : pixel_spacing : ...
           period/2 + (grid_size-1)*pixel_spacing/2;
y_starts = fliplr(x_starts);  % y 方向从上到下（匹配矩阵行顺序）

% ==================== 创建所有柱体 ====================
cylinder_tags = {};
count = 1;
for i = 1:grid_size
    for j = 1:grid_size
        if binary_matrix(i,j) == 1
            cyl_tag = ['cyl_' num2str(count)];
            geom.feature.create(cyl_tag, 'Cylinder');
            geom.feature(cyl_tag).set('r', num2str(cylinder_radius));
            geom.feature(cyl_tag).set('h', num2str(top_thickness));
            geom.feature(cyl_tag).set('pos', {num2str(x_starts(j)), num2str(y_starts(i)), '4e-7'});  % z位置放在绝缘层上
            geom.feature(cyl_tag).set('axis', {'0', '0', '1'});  % 沿 z 轴
            cylinder_tags{end+1} = cyl_tag;
            count = count + 1;
        end
    end
end

% ==================== 合并所有柱体（相邻自动连起来） ====================
if ~isempty(cylinder_tags)
    union_tag = 'top_union';
    geom.feature.create(union_tag, 'Union');
    geom.feature(union_tag).selection('input').set(cylinder_tags);
    geom.feature(union_tag).set('keep', 'off');  % 不保留内部边界 → 完全融合成一体
end

% ==================== 重建几何 ====================
geom.run;
mphsave(model, 'scripttest2.mph');