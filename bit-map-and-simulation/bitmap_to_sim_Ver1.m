

%加载基础模型
model = mphload('single.mph');
%网格参数
size=9;
size1=9;
size2=9;
size3=9;
size4=9;
size5=9;
%表面形状矩阵

%

% 删除柱体特征

% ==================== 参数设置 ====================
period = 2.8e-6;               % 器件边长 2.8 μm
top_thickness = 100e-9;        % 顶层厚度 100 nm
grid_size = 11;                % 11×11 网格
pixel_spacing = period / grid_size;  % 中心间距 ≈254.5 nm
cylinder_radius = pixel_spacing / 2 ;%selectable

z_base = 4e-7;                  % 柱体底部 z 坐标（根据你的模型调整）
z_center = z_base + top_thickness / 2;  % Block 中心 z 坐标




% 设置波长范围（单位：米）
lambda_start = 3;  
lambda_stop  = 12;  
lambda_step  = 5e-2;   

% ------------------ 你的 11×11 二值矩阵 ------------------
binary_matrix =[
    0 0 0 0 0 0 0 0 0 0 0;
    0 1 1 0 0 0 0 0 1 1 0;
    0 1 1 1 0 0 0 1 1 1 0;
    0 0 1 1 1 0 1 1 1 0 0;
    0 0 0 1 1 1 1 1 0 0 0;
    0 0 0 0 1 1 1 0 0 0 0;
    0 0 0 1 1 1 1 1 0 0 0;
    0 0 1 1 1 0 1 1 1 0 0;
    0 1 1 1 0 0 0 1 1 1 0;
    0 1 1 0 0 0 0 0 1 1 0;
    0 0 0 0 0 0 0 0 0 0 0];






%加载5000训练集
load('training_patterns_11x11.mat', 'selected');
N = size(selected,3);

% 预分配存储 S11 和 S21，尺寸为 [N, N_lambda]
lambda_vec = lambda_start:lambda_step:lambda_stop;
N_lambda = numel(lambda_vec);

% 预分配存储反射率（S11）和透射率（S21）
S11_all = complex(zeros(N, N_lambda));
S21_all = complex(zeros(N, N_lambda));

% 设置保存文件名
savefile = 'Sparams_dataset.mat';




%设置结束
function [S11,S21]=runsim(lambda_step,lambda_stop,lambda_start,binary_matrix,period,top_thickness,grid_size,pixel_spacing,cylinder_radius,z_base,z_center)
    
    geom = model.geom('geom1');
    geom.feature.remove('cyl1');
    wp_tag = 'wp1';
    
    wp = geom.feature.create(wp_tag, 'WorkPlane');
    wp.set('planetype', 'quick');
    wp.set('quickplane', 'xy');
    wp.set('quickz', z_base);  % 工作平面位于方块层底部
    
    wp = geom.feature(wp_tag);
    
    
    
    
    % ------------------ 在工作平面上生成选择性方块（Square） ------------------
    % 计算整个周期区域左下角坐标，使网格整体居中于原点
    x_min = -pixel_spacing/2;
    y_min = -pixel_spacing/2;
    
    square_tags = {};  % 收集所有方块标签
    
    for i = 1:grid_size
        for j = 1:grid_size
            if binary_matrix(i,j) == 1
                x_center = x_min + (j - 0.5) * pixel_spacing;
                y_center = y_min + (grid_size - i + 0.5) * pixel_spacing;  % y轴翻转
                
                feat_tag = sprintf('sq_%d_%d', i, j);
                
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
    
    % ------------------ 运行一次 2D 几何（生成所有方块） ------------------
    wp.geom().run;
    
    % ------------------ 对所有方块进行并集，合并为最少域 ------------------
    union2d_tag = 'un1';
    
    % 先删除可能存在的旧并集
    if any(strcmp(wp.geom().feature().tags, union2d_tag))
        wp.geom().feature().remove(union2d_tag);
    end
    
    % 创建并集特征
    un2d = wp.geom().feature().create(union2d_tag, 'Union');
    un2d.selection('input').set(square_tags);
    un2d.set('keep', false);  % <--- 关键：不保留原始方块，只保留并集结果
    un2d.set('intbnd', false);%删除内部边界原代码un2d.set('keepintbnd',fanlse)
    % 再次运行工作平面几何（执行并集，删除原始方块）
    wp.geom().run;
    
    % ------------------ 拉伸（现在工作平面只输出合并后的域） ------------------
    ext_tag = 'ext1';
    
    % 删除旧挤出（如存在）
    if any(strcmp(geom.feature.tags, ext_tag))
        geom.feature.remove(ext_tag);
    end
    
    ext = geom.feature.create(ext_tag, 'Extrude');
    ext.selection('input').set({wp_tag});     % 输入整个工作平面（已合并）
    ext.set('distance', top_thickness);
    
    geom.run;
    % ------------------ 可视化与保存 ------------------
    %mphgeom(model);             % 3D 视图
    %mphnavigator(model);        % 查看模型树，检查所有特征
    
    mesh = model.mesh('mesh1');
    %%设置网格大小
    %mesh.feature('size').set('hauto',size);%设置全局网格大小。
    %mesh.feature('size1').set('hauto',size1);
    %mesh.feature('size2').set('hauto'
    % ,size2);
    %mesh.feature('size3').set('hauto',size3);
    %mesh.feature('size4').set('hauto',size4);
    %mesh.feature('size5').set('hauto',size5);
    
    mesh.run;
    
    
    
    % 生成波长向量 (SI 单位：米)
    lambda_vec = lambda_start : lambda_step : lambda_stop;
    
    % 直接修改波长域特征
    model.study("std1").feature("wave").set("plist", lambda_vec);
    
    
    %运行模型
    study_tag = 'std1';
    study = model.study(study_tag);
    study.run;
    
    %提取参数
    S11 = mphglobal(model, 'ewfd.S1x', 'complexout', 'on');  % 反射系数（复数）
    S21 = mphglobal(model, 'ewfd.S2x', 'complexout', 'on');  % 透射系数（复数，如果无端口2则报错，可捕获）
    R = abs(S11).^2;  % 反射率 |S11|^2
    T = abs(S21).^2;  % 透射率 |S21|^2
    A = 1 - R - T;    % 吸收率（能量守恒）
    
    
   
    %mphsave(model, 'model_selective_squares_only.mph');
    
end

if isfile(savefile)
    load(savefile, 'S11_all', 'S21_all', 'i');
    start_i = i + 1;  % 从上次保存的进度开始
    fprintf('🔁 从第 %d 个样本继续运行\n', start_i);
else
    start_i = 1;  % 如果没有保存文件，从头开始
end

for i = start_i:N
    binary_matrix = selected(:,:,i);

    [S11,S21]=runsim(lambda_step,lambda_stop,lambda_start,binary_matrix,period,top_thickness,grid_size,pixel_spacing,cylinder_radius,z_base,z_center);

     S11_all(i,:) = S11(:).';
    S21_all(i,:) = S21(:).';

    % 每运行50个样本保存一次数据，以防止内存溢出
    if mod(i, 50) == 0 || i == N
        save(savefile, 'S11_all', 'S21_all', 'lambda_vec', 'i', '-v7.3');
        fprintf('已保存第 %d 个样本的数据\n', i);
    end

    % 打印当前进度
    fprintf('正在运行第 %d 个样本 / 共 %d 个样本\n', i, N);
end

lambda = lambda_start : lambda_step : lambda_stop;







R = abs(S11).^2;  % 反射率 |S11|^2
T = abs(S21).^2;  % 透射率 |S21|^2
A = 1 - R - T;    % 吸收率（能量守恒）
    
    

mphclose(model);
% 保存模型（可选）
%mphnavigator(model);