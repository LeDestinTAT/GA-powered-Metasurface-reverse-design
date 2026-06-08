%% =========================================================
%  单个 11x11 结构：提取整个域内的复电磁场数据
%  输出：
%    Ex_vol, Ey_vol, Ez_vol, Hx_vol, Hy_vol, Hz_vol
%  尺寸：
%    [N_lambda, Nx, Ny, Nz]
%  说明：
%    - x/y 方向：整个 period × period 区域均匀采样
%    - z 方向：整个域覆盖，并在关键界面与顶部结构附近加密
%    - 为避免内存过大，按波长逐个提取并写入 mat 文件
%% =========================================================

clear; clc;

%% ==================== 加载模型 ====================
model = mphload('single.mph');
geom  = model.geom('geom1');

%% ==================== 参数设置 ====================
period  = 2.8e-6;     % 2.8 um
r       = 0.7e-6;     %#ok<NASGU>  % 这里不直接用于方块生成，只保留参数
r_depth = 100e-9;     % 底金属厚度 100 nm
i_depth = 300e-9;     % 介质厚度 300 nm
s_depth = 30e-9;      % 顶部图案厚度 30 nm

grid_size      = 11;                 % 11x11
pixel_spacing  = period / grid_size; % 单元像素尺寸
top_thickness  = s_depth;            % 顶部图案真实厚度
z_base         = r_depth + i_depth;  % 顶部图案底面 z = 400 nm

% ---------------- x/y 采样密度（可改） ----------------
% 你这次只明确要求了 z 向分布，因此 x/y 我保留为可调参数
Nx = 51;
Ny = 51;

% ---------------- 保存文件名 ----------------
savefile = 'field_dataset_single_sample.mat';
if exist(savefile, 'file')
    delete(savefile);
end

%% ==================== 删除旧特征（避免重复运行冲突） ====================
if any(strcmp(geom.feature.tags, 'cyl1'))
    geom.feature.remove('cyl1');
end
if any(strcmp(geom.feature.tags, 'ext1'))
    geom.feature.remove('ext1');
end
if any(strcmp(geom.feature.tags, 'wp1'))
    geom.feature.remove('wp1');
end

%% ==================== 创建工作平面 ====================
wp_tag = 'wp1';
wp = geom.feature.create(wp_tag, 'WorkPlane');
wp.set('planetype', 'quick');
wp.set('quickplane', 'xy');
wp.set('quickz', z_base);

wp = geom.feature(wp_tag);

%% ==================== 11x11 二值矩阵 ====================
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
    0 0 0 0 0 0 0 0 0 0 0
];

%% ==================== 在工作平面上生成选择性方块 ====================
% 你的域在 x,y 上是 [0, period] × [0, period]
x_min = 0;
y_min = 0;

square_tags = {};

for i = 1:grid_size
    for j = 1:grid_size
        if binary_matrix(i,j) == 1
            x_center = x_min + (j - 0.5) * pixel_spacing;
            y_center = y_min + (grid_size - i + 0.5) * pixel_spacing;  % y 方向翻转

            feat_tag = sprintf('sq_%d_%d', i, j);

            sq = wp.geom().feature().create(feat_tag, 'Square');
            sq.set('size', pixel_spacing);
            sq.set('pos', [x_center, y_center]);

            square_tags{end+1} = feat_tag; %#ok<SAGROW>
        end
    end
end

if isempty(square_tags)
    error('二值矩阵全为 0，没有方块需要生成。');
end

%% ==================== 运行 2D 几何 ====================
wp.geom().run;

%% ==================== 并集 ====================
union2d_tag = 'un1';

if any(strcmp(wp.geom().feature().tags, union2d_tag))
    wp.geom().feature().remove(union2d_tag);
end

un2d = wp.geom().feature().create(union2d_tag, 'Union');
un2d.selection('input').set(square_tags);
un2d.set('keep', false);
un2d.set('intbnd', false);

wp.geom().run;

%% ==================== 拉伸为 3D 顶部图案 ====================
ext_tag = 'ext1';
ext = geom.feature.create(ext_tag, 'Extrude');
ext.selection('input').set({wp_tag});
ext.set('distance', top_thickness);

geom.run;

%% ==================== 网格 ====================
mesh = model.mesh('mesh1');
mesh.run;

%% ==================== 求解 ====================
study_tag = 'std1';
study = model.study(study_tag);
study.run;

%% ==================== 读取波长与 S 参数参考 ====================
lambda = mphglobal(model, 'lambda0', 'complexout', 'off');
lambda = lambda(:);
N_lambda = numel(lambda);

S11_ref = mphglobal(model, 'ewfd.S1x', 'complexout', 'on');
S21_ref = mphglobal(model, 'ewfd.S2x', 'complexout', 'on');

R_ref = abs(S11_ref).^2;
T_ref = abs(S21_ref).^2;
A_ref = 1 - R_ref - T_ref;

%% =========================================================
%  定义整个域的空间采样
%  x/y：均匀覆盖整个域 [0, period]
%  z：按你给的结构信息分段加密
%
%  域结构：
%    0 ~ 100 nm     : 底金属层
%    100 ~ 400 nm   : 介质层
%    400 ~ 430 nm   : 顶部图案层
%    430 ~ 6400 nm  : 上方空气域
%
%  重点加密：
%    1) z = 100 nm 附近（金属/介质界面）
%    2) z = 370 ~ 460 nm（顶部结构上下各一倍厚度范围）
%% =========================================================

% x/y 坐标
xv = linspace(0, period, Nx);
yv = linspace(0, period, Ny);

% z 坐标分段
z_metal_bottom = 0;
z_metal_top    = r_depth;                % 100 nm
z_diel_top     = r_depth + i_depth;      % 400 nm
z_struct_bot   = z_diel_top;             % 400 nm
z_struct_top   = z_struct_bot + s_depth; % 430 nm
z_air_top      = z_struct_bot + 6e-6;    % 6.4 um

% 底金属到界面前：较疏
z_seg_1 = linspace(z_metal_bottom, 70e-9, 8);

% z = 100 nm 界面附近：次级加密
z_seg_2 = linspace(70e-9, 130e-9, 21);

% 介质主体：中等密度
z_seg_3 = linspace(130e-9, 370e-9, 21);

% 顶部图案层附近主加密区：370 ~ 460 nm
z_seg_4 = linspace(370e-9, 400e-9, 19);  % 下方一倍厚度
z_seg_5 = linspace(400e-9, 430e-9, 25);  % 图案本体最密
z_seg_6 = linspace(430e-9, 460e-9, 19);  % 上方一倍厚度

% 上方空气近场区：中等密度
z_seg_7 = linspace(460e-9, 1.0e-6, 28);

% 上方空气远区：较疏
z_seg_8 = linspace(1.0e-6, z_air_top, 36);

zv = unique([z_seg_1, z_seg_2, z_seg_3, z_seg_4, z_seg_5, z_seg_6, z_seg_7, z_seg_8]);
zv = sort(zv(:).');
Nz = numel(zv);

fprintf('总波长点数 N_lambda = %d\n', N_lambda);
fprintf('空间采样尺寸 = [%d, %d, %d]\n', Nx, Ny, Nz);
fprintf('z 向范围 = [%.3f, %.3f] um\n', zv(1)*1e6, zv(end)*1e6);

%% ==================== 构造体采样坐标 ====================
[Xv, Yv, Zv] = ndgrid(xv, yv, zv);
coord_vol = [Xv(:).'; Yv(:).'; Zv(:).'];

N_pts = numel(Xv);
fprintf('总采样点数 = %d\n', N_pts);

%% ==================== 先保存元数据 ====================
save(savefile, ...
    'binary_matrix', ...
    'period', 'r_depth', 'i_depth', 's_depth', ...
    'grid_size', 'pixel_spacing', 'top_thickness', 'z_base', ...
    'Nx', 'Ny', 'Nz', 'xv', 'yv', 'zv', ...
    'lambda', ...
    'S11_ref', 'S21_ref', 'R_ref', 'T_ref', 'A_ref', ...
    '-v7.3');

%% ==================== 创建 matfile 并预分配磁盘变量 ====================
mf = matfile(savefile, 'Writable', true);

% 预分配为 complex single，减少磁盘与内存压力
mf.Ex_vol(N_lambda, Nx, Ny, Nz) = complex(single(0));
mf.Ey_vol(N_lambda, Nx, Ny, Nz) = complex(single(0));
mf.Ez_vol(N_lambda, Nx, Ny, Nz) = complex(single(0));
mf.Hx_vol(N_lambda, Nx, Ny, Nz) = complex(single(0));
mf.Hy_vol(N_lambda, Nx, Ny, Nz) = complex(single(0));
mf.Hz_vol(N_lambda, Nx, Ny, Nz) = complex(single(0));

%% ==================== 按波长逐个提取场并写盘 ====================
exprs = {'ewfd.Ex','ewfd.Ey','ewfd.Ez','ewfd.Hx','ewfd.Hy','ewfd.Hz'};

for k = 1:N_lambda
    fprintf('提取第 %d / %d 个波长点 ...\n', k, N_lambda);

    [Ex_k, Ey_k, Ez_k, Hx_k, Hy_k, Hz_k] = mphinterp( ...
        model, exprs, ...
        'coord',      coord_vol, ...
        'edim',       'domain', ...
        'solnum',     k, ...
        'complexout', 'on', ...
        'coorderr',   'on' );

    % reshape 回 [Nx, Ny, Nz]
    Ex_k = reshape(single(Ex_k), [1, Nx, Ny, Nz]);
    Ey_k = reshape(single(Ey_k), [1, Nx, Ny, Nz]);
    Ez_k = reshape(single(Ez_k), [1, Nx, Ny, Nz]);
    Hx_k = reshape(single(Hx_k), [1, Nx, Ny, Nz]);
    Hy_k = reshape(single(Hy_k), [1, Nx, Ny, Nz]);
    Hz_k = reshape(single(Hz_k), [1, Nx, Ny, Nz]);

    % 写入磁盘
    mf.Ex_vol(k, :, :, :) = Ex_k;
    mf.Ey_vol(k, :, :, :) = Ey_k;
    mf.Ez_vol(k, :, :, :) = Ez_k;
    mf.Hx_vol(k, :, :, :) = Hx_k;
    mf.Hy_vol(k, :, :, :) = Hy_k;
    mf.Hz_vol(k, :, :, :) = Hz_k;
end

%% ==================== 可选：简单画一个切片检查 ====================
% 取中间一个波长、顶部结构附近一个 z 层看 |Ez|
k_plot = round(N_lambda/2);

[~, iz_plot] = min(abs(zv - (z_struct_bot + 0.5*s_depth)));

Ez_slice = squeeze(mf.Ez_vol(k_plot, :, :, iz_plot));

figure('Name', '中部结构层附近 |Ez|');
imagesc(xv*1e6, yv*1e6, abs(Ez_slice).');
set(gca, 'YDir', 'normal');
axis image;
xlabel('x (um)');
ylabel('y (um)');
title(sprintf('|Ez| at lambda = %.4f um, z = %.1f nm', ...
    lambda(k_plot)*1e6, zv(iz_plot)*1e9));
colorbar;

%% ==================== 可选：S 参数参考图 ====================
figure('Name', 'S 参数参考');
plot(lambda*1e6, R_ref, 'b-', 'LineWidth', 1.8); hold on;
plot(lambda*1e6, T_ref, 'g--', 'LineWidth', 1.8);
plot(lambda*1e6, A_ref, 'r-', 'LineWidth', 1.8);
plot(lambda*1e6, R_ref + T_ref + A_ref, 'k:', 'LineWidth', 1.2);
xlabel('\lambda (um)');
ylabel('Response');
legend('R','T','A','R+T+A');
grid on;

%% ==================== 保存模型（可选） ====================
mphsave(model, 'model_selective_squares_field_sampling_final.mph');

fprintf('完成。场数据已保存到: %s\n', savefile);