% 加载5000训练集
load('training_patterns_11x11.mat', 'selected');
N = size(selected, 3);  % 样本数量

% 预分配存储 S11 和 S21，尺寸为 [N, N_lambda]
lambda_start = 3;  
lambda_stop  = 12;  
lambda_step  = 5e-2;
lambda_vec = lambda_start : lambda_step : lambda_stop;
N_lambda = numel(lambda_vec);

% 预分配存储反射率（S11）和透射率（S21）
S11_all = complex(zeros(N, N_lambda));
S21_all = complex(zeros(N, N_lambda));

% 设置保存文件名
savefile = 'Sparams_dataset.mat';

% ==================== 参数设置 ====================
period = 2.8e-6;               % 器件边长 2.8 μm
top_thickness = 100e-9;        % 顶层厚度 100 nm
grid_size = 11;                % 11×11 网格
pixel_spacing = period / grid_size;  % 中心间距 ≈254.5 nm
cylinder_radius = pixel_spacing / 2 ; % 可调

z_base = 4e-7;                  % 柱体底部 z 坐标（根据你的模型调整）
z_center = z_base + top_thickness / 2;  % Block 中心 z 坐标

% ==================== 主循环 ====================
for i = 1:N
    % 获取当前的二值矩阵
    binary_matrix = selected(:,:,i);

    % 1. 加载 COMSOL 模型
    model = mphload('single.mph');
    geom = model.geom('geom1');
    geom.feature.remove('cyl1');
    
    wp_tag = 'wp1';
    wp = geom.feature.create(wp_tag, 'WorkPlane');
    wp.set('planetype', 'quick');
    wp.set('quickplane', 'xy');
    wp.set('quickz', z_base);  % 工作平面位于方块层底部
    wp = geom.feature(wp_tag);

    % 2. 在工作平面上生成选择性方块（Square）
    x_min = -pixel_spacing / 2;
    y_min = -pixel_spacing / 2;
    square_tags = {};  % 收集所有方块标签

    for j = 1:grid_size
        for k = 1:grid_size
            if binary_matrix(j, k) == 1
                x_center = x_min + (k - 0.5) * pixel_spacing;
                y_center = y_min + (grid_size - j + 0.5) * pixel_spacing;  % y轴翻转

                feat_tag = sprintf('sq_%d_%d', j, k);
                sq = wp.geom().feature().create(feat_tag, 'Square');
                sq.set('size', pixel_spacing);
                sq.set('pos', [x_center, y_center]);
                square_tags{end+1} = feat_tag;
            end
        end
    end

    % 如果没有填充，直接退出（避免空并集报错）
    if isempty(square_tags)
        error('二值矩阵全为0，没有方块需要生成');
    end

    % 3. 运行一次 2D 几何（生成所有方块）
    wp.geom().run;

    % 4. 对所有方块进行并集，合并为最少域
    union2d_tag = 'un1';
    if any(strcmp(wp.geom().feature().tags, union2d_tag))
        wp.geom().feature().remove(union2d_tag);
    end
    un2d = wp.geom().feature().create(union2d_tag, 'Union');
    un2d.selection('input').set(square_tags);
    un2d.set('keep', false);  % 不保留原始方块，只保留并集结果
    un2d.set('intbnd', false);  % 删除内部边界
    wp.geom().run;

    % 5. 拉伸（工作平面只输出合并后的域）
    ext_tag = 'ext1';
    if any(strcmp(geom.feature.tags, ext_tag))
        geom.feature.remove(ext_tag);
    end
    ext = geom.feature.create(ext_tag, 'Extrude');
    ext.selection('input').set({wp_tag});  % 输入整个工作平面（已合并）
    ext.set('distance', top_thickness);
    geom.run;

    % 6. 设置网格并运行仿真
    mesh = model.mesh('mesh1');
    mesh.run;

    % 7. 生成波长向量 (SI 单位：米)
    lambda_vec = lambda_start : lambda_step : lambda_stop;

    % 8. 设置波长范围
    model.study("std1").feature("wave").set("plist", lambda_vec);

    % 9. 运行模型
    study_tag = 'std1';
    study = model.study(study_tag);
    study.run;

    % 10. 提取反射率和透射率
    S11 = mphglobal(model, 'ewfd.S1x', 'complexout', 'on');  % 反射系数（复数）
    S21 = mphglobal(model, 'ewfd.S2x', 'complexout', 'on');  % 透射系数（复数）
    R = abs(S11).^2;  % 反射率 |S11|^2
    T = abs(S21).^2;  % 透射率 |S21|^2
    A = 1 - R - T;    % 吸收率（能量守恒）

    % 存储 S11 和 S21
    S11_all(i, :) = S11(:).';
    S21_all(i, :) = S21(:).';

    % 11. 每50个样本保存一次数据，以防止内存溢出
    if mod(i, 50) == 0 || i == N
        save(savefile, 'S11_all', 'S21_all', 'lambda_vec', 'i', '-v7.3');
        fprintf('已保存第 %d 个样本的数据\n', i);
    end

    % 打印当前进度
    fprintf('正在运行第 %d 个样本 / 共 %d 个样本\n', i, N);

    % 12. 关闭 COMSOL 模型
    clear model;
end

% 计算反射率和透射率
R = abs(S11_all).^2;  % 反射率 |S11|^2
T = abs(S21_all).^2;  % 透射率 |S21|^2
A = 1 - R - T;        % 吸收率（能量守恒）

% 保存最终数据
save('dataset_for_FNO.mat', 'S11_all', 'S21_all', 'A', 'lambda_vec', '-v7.3');
 