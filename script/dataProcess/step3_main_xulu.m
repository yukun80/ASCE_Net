% =========================================================================
% 脚本名称: step3_main_xulu.m
% 功能概述: 批量读取输入目录中的.raw雷达成像数据，提取属性散射中心并保存；
%           基于提取的散射中心进行图像重建，同时保存重建结果、差异图以及JPEG可视化。
%
% 输入目录:
%   - project_root/02_Data_Processed_raw  (递归遍历其所有子目录下的 .raw 文件)
%
% 输出目录:
%   - project_root/03_Training_ASC                 (保存散射中心 .mat，变量: scatter_all)
%   - project_root/03_Training_ASC_reconstruct     (保存重建结果与差异 .mat，变量: s, diff)
%   - project_root/03_Training_ASC_reconstruct_jpeg(保存重建结果 JPEG 可视化)
%
% 使用方法:
%   - 在脚本中配置 project_root（见“用户配置”部分），确保输入/输出目录存在或可创建；
%   - 直接运行脚本，程序将递归处理全部 .raw 文件并保持目录层级结构；
%   - 处理进度与当前文件路径将输出到命令行窗口。
%
% 依赖函数:
%   - image_read(raw_filepath) -> [fileimage, image_value]
%   - extrac(fileimage, image_value) -> scatter_all
%   - simulation(scatter_all) -> s
%
% 注意事项:
%   - 本脚本会对输出目录按输入的相对路径自动创建子目录；
%   - JPEG 保存前会对重建结果 s 进行 mat2gray 归一化；
%   - 请确保依赖函数位于 MATLAB 搜索路径中；
%   - 如需变更数据路径或导出格式，请修改“用户配置”区和相关保存语句。
% =========================================================================

clear;
clc;
close all;

% --- 用户配置 ---
% 定义根目录，方便切换整个项目的位置
project_root = 'E:\Document\paper_library\3rd_paper_250512\code\ASCE_Net\datasets\SAR_ASC_Project\';

% 输入目录：包含.raw文件的目录
input_dir = fullfile(project_root, '02_Data_Processed_raw');

% 输出目录
output_asc_dir = fullfile(project_root, '03_Training_ASC'); % 存放散射中心
output_recons_dir = fullfile(project_root, '03_Training_ASC_reconstruct'); % 存放重建结果
output_jpeg_dir = fullfile(project_root, '03_Training_ASC_reconstruct_jpeg'); % 存放JPEG格式的重建图

% --- 主程序 ---
% 检查输入目录是否存在
if ~isfolder(input_dir)
    error('输入目录不存在: %s', input_dir);
end

% 递归查找所有.raw文件
fprintf('正在从 %s 中查找.raw文件...\n', input_dir);
files = dir(fullfile(input_dir, '**', '*.raw'));
fprintf('找到了 %d 个.raw文件。\n', length(files));

for i = 1:length(files)
    input_raw_file = fullfile(files(i).folder, files(i).name);
    fprintf('正在处理文件 (%d/%d): %s\n', i, length(files), input_raw_file);
    
    % --- 1. 构建输出路径并创建文件夹 ---
    % 获取相对路径
    relative_path = erase(files(i).folder, input_dir);
    
    % 创建散射中心输出目录
    current_output_asc_dir = fullfile(output_asc_dir, relative_path);
    if ~isfolder(current_output_asc_dir)
        mkdir(current_output_asc_dir);
    end
    
    % 创建重建结果输出目录
    current_output_recons_dir = fullfile(output_recons_dir, relative_path);
    if ~isfolder(current_output_recons_dir)
        mkdir(current_output_recons_dir);
    end
    
    % 创建JPEG重建结果输出目录
    current_output_jpeg_dir = fullfile(output_jpeg_dir, relative_path);
    if ~isfolder(current_output_jpeg_dir)
        mkdir(current_output_jpeg_dir);
    end
    
    % 获取不带扩展名的文件名
    [~, basename, ~] = fileparts(files(i).name);

    % --- 2. 读取和处理图像 ---
    [fileimage, image_value] = image_read(input_raw_file);
    
    % 算法1主函数：提取散射中心
    scatter_all = extrac(fileimage, image_value);
    
    % 保存提取的属性散射中心
    asc_output_filename = [basename, '_yang.mat'];
    save(fullfile(current_output_asc_dir, asc_output_filename), 'scatter_all');
        
    % --- 3. 结果重建和保存 ---
    s = simulation(scatter_all); % 图1：重建图
    diff = fileimage - s;
    
    % 保存重建的切片和差异 (.mat格式)
    recons_output_filename = [basename, '_yangRecon.mat'];
    save(fullfile(current_output_recons_dir, recons_output_filename), 's', 'diff');
    
    % 将重建图像归一化并保存为JPEG
    s_normalized = mat2gray(s); % 将矩阵s归一化到[0, 1]范围
    jpeg_output_filename = [basename, '_reconstruct.jpg'];
    imwrite(s_normalized, fullfile(current_output_jpeg_dir, jpeg_output_filename));
    
    % 关闭所有打开的图像窗口，避免弹出大量窗口
    close all;
end

fprintf('所有文件处理完毕。\n');

