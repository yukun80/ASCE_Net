%% ========================================================================
%  SAR 特征提取结果可视化脚本 (visualize_results.m)
%  版本 v2.0 - 根据新的文件命名规则和路径结构进行修改
% =========================================================================
clear all;
close all;
clc;

% --- 用户配置区 ---
% 请在这里指定您要可视化的文件基础名
% 例如，对于 "HB03333.015.128x128.raw"，这里应填写 'HB03333.015'
filename_base = 'HB03344.015'; % <--- 修改这里

% 定义数据路径
original_data_path = 'E:\Document\_Mission\2025\250512_三维目标散射机理研究与特征提取研究\MSTAR数据集\算法1_杨老师\datasourceProcess\Step1_SourceToYang\';
output_path = 'E:\Document\_Mission\2025\250512_三维目标散射机理研究与特征提取研究\MSTAR数据集\算法1_杨老师\datasourceProcess\Step2_OutputofYang\';

% --- 文件名和路径构建 ---
% 根据基础名自动构建完整的文件名
original_raw_filename = [filename_base, '.128x128.raw'];
results_mat_filename = [filename_base, '_yang.mat'];

original_raw_filepath = fullfile(original_data_path, original_raw_filename);
results_mat_filepath = fullfile(output_path, results_mat_filename);


% --- 数据加载区 ---
fprintf('正在加载数据: %s ...\n', filename_base);

% 1. 加载原始SAR复数数据 (使用 image_read.m)
if ~exist(original_raw_filepath, 'file')
    error('找不到原始数据文件: %s', original_raw_filepath);
end
% 注意：这里的原始数据需要用 image_read 来正确解析幅度和相位
[fileimage_original, image_value_original] = image_read(original_raw_filepath);

% 2. 加载提取的散射中心参数
if ~exist(results_mat_filepath, 'file')
    error('找不到结果文件: %s. \n请先运行 step2_main_xulu.m 来生成结果。', results_mat_filepath);
end
load(results_mat_filepath, 'scatter_all'); % 加载 scatter_all 变量


% --- 可视化执行区 ---

%% 1. 特征参数可视化 (控制台表格)
fprintf('\n================== 提取的散射中心参数 (%s) ==================\n', filename_base);
fprintf('编号 |   类型   |   幅度(A)  | X坐标(m) | Y坐标(m) |  长度(L) | 倾斜角(φ'') | 频率依赖(α) \n');
fprintf('-------------------------------------------------------------------------------------------\n');

if isempty(scatter_all)
    fprintf('未提取到任何散射中心。\n');
else
    for i = 1:length(scatter_all)
        params = scatter_all{i};
        x = params(1);
        y = params(2);
        alpha = params(3);
        % gamma = params(4); % gamma/r 依赖, 在此模型中不突出，暂不显示
        phi_prime = params(5);
        L = params(6);
        A = params(7);
        
        % 判断散射中心类型
        if L == 0 && phi_prime == 0
            type_str = '局部';
            fprintf('%-5d| %-9s| %-11.4f| %-9.4f| %-9.4f| %-9s| %-11s| %-12.2f\n', ...
                    i, type_str, A, x, y, 'N/A', 'N/A', alpha);
        else
            type_str = '分布';
            fprintf('%-5d| %-9s| %-11.4f| %-9.4f| %-9.4f| %-9.4f| %-11.4f| %-12.2f\n', ...
                    i, type_str, A, x, y, L, phi_prime, alpha);
        end
    end
end
fprintf('===========================================================================================\n\n');


%% 2. 图像域对比可视化
fprintf('正在生成图像域对比图...\n');

% 首先，根据提取的特征重建图像
image_reconstructed = simulation(scatter_all);

% 创建一个新的figure窗口
figure('Name', ['图像域结果对比 - ' filename_base], 'Position', [100, 300, 1400, 500]);

% 子图1: 原始幅度图
subplot(1, 3, 1);
imshow(fileimage_original, []);
title({'1. 原始SAR幅度图', '(Original SAR Image)'});
xlabel('距离向 (Range)');
ylabel('方位向 (Azimuth)');

% 子图2: 重建的散射中心图
subplot(1, 3, 2);
imshow(image_reconstructed, []);
title({'2. 重建的散射中心图', '(Reconstructed from Features)'});
xlabel('距离向 (Range)');
ylabel('方位向 (Azimuth)');

% 子图3: 残差图 (差异)
residual_image = abs(fileimage_original - image_reconstructed);
subplot(1, 3, 3);
imshow(residual_image, []);
title({'3. 残差图 (Residual = Original - Reconstructed)'});
xlabel('距离向 (Range)');
ylabel('方位向 (Azimuth)');
colorbar; % 显示色阶条，以观察差异的量级

%% 3. 频域对比可视化
fprintf('正在生成频域对比图...\n');

% --- 为获取精确频域重建，复用simulation的核心逻辑 ---
fc=1e10; B=5e8; om=2.86; q=128;
freq_reconstructed_complex = zeros(q,q);
if ~isempty(scatter_all)
    for i = 1:length(scatter_all)
        params = scatter_all{i};
        % 调用spotlight函数生成每个散射中心的频域贡献
        [K_temp, ~] = spotlight(fc,B,om,params(1),params(2),params(3),params(4),params(5),params(6),params(7));
        freq_reconstructed_complex = freq_reconstructed_complex + K_temp;
    end
end
% --- 结束 ---

freq_original_complex = fftshift(fft2(image_value_original));

figure('Name', ['频域结果对比 - ' filename_base], 'Position', [150, 200, 1200, 500]);

% 子图1: 原始图像频域幅度谱
subplot(1, 2, 1);
mesh(log(abs(freq_original_complex) + 1e-9)); % 使用log增强显示效果, +1e-9避免log(0)
title({'原始图像频域幅度谱', '(Original Spectrum)'});
xlabel('方位频率');
ylabel('距离频率');
zlabel('Log(幅度)');
view(30, 45); % 调整视角

% 子图2: 重建图像频域幅度谱
subplot(1, 2, 2);
mesh(log(abs(freq_reconstructed_complex) + 1e-9));
title({'重建图像频域幅度谱', '(Reconstructed Spectrum)'});
xlabel('方位频率');
ylabel('距离频率');
zlabel('Log(幅度)');
view(30, 45);

fprintf('所有可视化已完成。\n');