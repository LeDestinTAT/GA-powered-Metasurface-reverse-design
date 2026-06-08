% 加载模型
model = mphload('single.mph');

geom = model.geom('geom1');  % 你的几何序列 tag

% 删除旧柱体特征（如果存在）

geom.feature.remove('cyl1');


% ==================== 参数设置 ====================
period = 2.8e-6;               % 器件边长 2.8 μm
top_thickness = 100e-9;        % 顶层厚度 100 nm
grid_size = 11;                % 11×11 网格
pixel_spacing = period / grid_size;  % 中心间距 ≈254.5 nm
cylinder_radius = pixel_spacing / 2 ;%selectable

z_base = 4e-7;                  % 柱体底部 z 坐标（根据你的模型调整）
z_center = z_base + top_thickness / 2;  % Block 中心 z 坐标

% ==================== 二值矩阵（'1' = 生成柱体） ====================
binary_matrix = [
    0 0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 1 1 1 1 1 0 0;
    0 0 0 1 1 1 1 1 1 1 0;
    0 0 1 1 1 1 1 1 1 1 0;
    0 1 1 1 1 1 1 1 1 1 0;
    0 1 1 1 1 1 1 1 1 1 0;
    0 1 1 1 1 1 1 1 1 1 0;
    0 1 1 1 1 1 1 1 1 0 0;
    0 0 1 1 1 1 1 1 0 0 0;
    0 0 0 1 1 1 1 0 0 0 0;
    0 0 0 0 0 0 0 0 0 0 0];

% ==================== 计算中心坐标（整体中心在 period/2, period/2） ====================
x_coords = linspace(pixel_spacing/2, period - pixel_spacing/2, grid_size);
y_coords = fliplr(x_coords);  % y 方向翻转匹配矩阵行

% ==================== 创建柱体 ====================
cylinder_tags = {};
count = 1;
for i = 1:grid_size
    for j = 1:grid_size
        if binary_matrix(i,j) == 1
            cyl_tag = ['cyl_' num2str(count)];
            geom.feature.create(cyl_tag, 'Cylinder');
            geom.feature(cyl_tag).set('r', num2str(cylinder_radius));
            geom.feature(cyl_tag).set('h', num2str(top_thickness));
            geom.feature(cyl_tag).set('pos', {num2str(x_coords(j)), num2str(y_coords(i)), num2str(z_base)});
            geom.feature(cyl_tag).set('axis', {'0', '0', '1'});
            cylinder_tags{end+1} = cyl_tag;
            count = count + 1;
        end
    end
end

% ==================== 填充相邻柱体间隙（桥接） ====================
bridge_tags = {};
bridge_count = 1;

% 水平桥（x 方向填满）
for i = 1:grid_size
    for j = 1:grid_size-1
        if binary_matrix(i,j) == 1 && binary_matrix(i,j+1) == 1
            bridge_tag = ['bridge_h_' num2str(bridge_count)];
            geom.feature.create(bridge_tag, 'Block');
            % size: x=间距（填满），y=直径（覆盖柱体），z=厚度
            geom.feature(bridge_tag).set('size', {num2str(pixel_spacing), num2str(cylinder_radius*2), num2str(top_thickness)});
            bridge_x = (x_coords(j) + x_coords(j+1)-pixel_spacing) / 2;
            bridge_y = y_coords(i)-pixel_spacing/2;
            geom.feature(bridge_tag).set('pos', {num2str(bridge_x), num2str(bridge_y), num2str(z_center)});
            bridge_tags{end+1} = bridge_tag;
            bridge_count = bridge_count + 1;
        end
    end
end

% 垂直桥（y 方向填满）
for j = 1:grid_size
    for i = 1:grid_size-1
        if binary_matrix(i,j) == 1 && binary_matrix(i+1,j) == 1
            bridge_tag = ['bridge_v_' num2str(bridge_count)];
            geom.feature.create(bridge_tag, 'Block');
            % size: x=直径，y=间距（填满），z=厚度
            geom.feature(bridge_tag).set('size', {num2str(cylinder_radius*2), num2str(pixel_spacing), num2str(top_thickness)});
            bridge_x = x_coords(j)-pixel_spacing/2;
            bridge_y = (y_coords(i) + y_coords(i+1)-pixel_spacing) / 2;
            geom.feature(bridge_tag).set('pos', {num2str(bridge_x), num2str(bridge_y), num2str(z_center)});
            bridge_tags{end+1} = bridge_tag;
            bridge_count = bridge_count + 1;
        end
    end
end

% ==================== 合并所有对象（柱体 + 桥） ====================
all_tags = [cylinder_tags, bridge_tags];
if ~isempty(all_tags)
    union_tag = 'top_union';
    geom.feature.create(union_tag, 'Union');
    geom.feature(union_tag).selection('input').set(all_tags);
    geom.feature(union_tag).set('keep', 'off');  % 完全融合，无内部边界
end

% ==================== 重建几何 ====================
geom.run;

mat = model.material;

gold_tag = 'mat2';  % ← 替换为你的金材料实际 tag，例如 'Au' 或 'Gold'


