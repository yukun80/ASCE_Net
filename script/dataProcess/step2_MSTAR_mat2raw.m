% =============================================================================
% step2_MSTAR_mat2raw.m
% =============================================================================
%
% 功能:
% 该脚本自动将 '01_Data_Processed_mat' 目录中的所有 .mat 文件转换为
% .raw 二进制格式，同时保留原始的目录结构。转换后的 .raw 文件
% 将被保存在 '02_Data_Processed_raw' 目录中。此外，为了方便快速
% 可视化检查，脚本还会为每个 .raw 文件生成一个 .JPG 预览图像，
% 并将其存放在 '03_Data_Processed_jpg' 目录中，同样保持目录结构。
%
% =============================================================================

clear;
clc;

% --- 配置 ---
% 获取当前脚本所在文件夹的路径，并由此确定项目根目录
try
    [scriptFolder, ~, ~] = fileparts(mfilename('fullpath'));
    % 从脚本文件夹返回到项目根目录
    projectRoot = fullfile(scriptFolder, '..', '..');
catch
    % 如果在实时编辑器或无文件上下文的环境中运行，则使用当前工作目录
    projectRoot = pwd;
    warning('无法使用 mfilename 确定路径，将使用当前工作目录作为项目根目录。');
end

% 设置源路径和目标路径
input_dir = fullfile(projectRoot, 'datasets', 'SAR_ASC_constract', '01_Data_Processed_mat');
output_dir_raw = fullfile(projectRoot, 'datasets', 'SAR_ASC_constract', '02_Data_Processed_raw');
output_dir_jpg = fullfile(projectRoot, 'datasets', 'SAR_ASC_constract', '02_Data_Processed_jpg_tmp');


% --- 主执行逻辑 ---
fprintf('开始处理...\n');
fprintf('源目录: %s\n', input_dir);
fprintf('目标 RAW 目录: %s\n', output_dir_raw);
fprintf('目标 JPG 目录: %s\n', output_dir_jpg);

% 检查源目录是否存在
if ~isfolder(input_dir)
    error('源目录不存在或不是一个文件夹: %s', input_dir);
end

% 创建目标根目录（如果它们不存在）
if ~isfolder(output_dir_raw)
    mkdir(output_dir_raw);
end
if ~isfolder(output_dir_jpg)
    mkdir(output_dir_jpg);
end

% 调用递归函数开始处理
process_directory(input_dir, output_dir_raw, output_dir_jpg);

fprintf('所有文件处理完成。\n');

% --- 辅助函数 ---

function process_directory(current_source_path, current_raw_path, current_jpg_path)
    % 获取当前源路径下的所有内容
    items = dir(current_source_path);
    
    for i = 1:length(items)
        item_name = items(i).name;
        
        % 忽略 '.' 和 '..' 目录以及隐藏文件
        if strcmp(item_name, '.') || strcmp(item_name, '..') || startsWith(item_name, '.')
            continue;
        end
        
        source_item_path = fullfile(current_source_path, item_name);
        target_item_raw_path = fullfile(current_raw_path, item_name);
        target_item_jpg_path = fullfile(current_jpg_path, item_name);

        if items(i).isdir
            % 如果是目录，则创建对应的目标目录并进行递归
            if ~isfolder(target_item_raw_path)
                mkdir(target_item_raw_path);
            end
            if ~isfolder(target_item_jpg_path)
                mkdir(target_item_jpg_path);
            end
            process_directory(source_item_path, target_item_raw_path, target_item_jpg_path);
        else
            % 如果是文件，并且是 .mat 文件，则进行处理
            [~, ~, ext] = fileparts(item_name);
            if ~strcmpi(ext, '.mat')
                continue; % 只处理 .mat 文件
            end

            fprintf('正在处理: %s\n', source_item_path);
            
            % 定义输出文件的基本名称 (不含扩展名)
            output_basename = item_name(1:end-4);
            
            % --- 步骤 1: 将 .mat 转换为 .raw ---
            output_base_path_raw = fullfile(current_raw_path, output_basename);
            % 先读取 Img 尺寸以构造对应的 RAW 文件名
            try
                tmp_info = load(source_item_path, 'Img');
                [rows, cols] = size(tmp_info.Img);
                raw_file_name = sprintf('%s.%dx%d.raw', output_base_path_raw, cols, rows);
            catch ME
                warning('  -> 读取 Img 尺寸失败: %s', ME.message);
                raw_file_name = '';
            end
            create_R1_for_image_read(source_item_path, output_base_path_raw);
            
            % --- 步骤 2: 创建 JPG 预览 ---
            
            if ~isempty(raw_file_name) && isfile(raw_file_name)
                [fileimage, ~] = image_read(raw_file_name);
                
                % 定义输出 JPG 文件的路径
                output_jpg_v1 = fullfile(current_jpg_path, [output_basename, '_v1.JPG']);
                output_jpg_v2 = fullfile(current_jpg_path, [output_basename, '_v2.JPG']);
                
                % 保存两个版本的 JPG 以便更好地可视化
                imwrite(uint8(imadjust(fileimage) * 255), output_jpg_v1); % 带对比度调整
                imwrite(uint8(fileimage / max(fileimage(:)) * 255), output_jpg_v2); % 简单归一化
                
                fprintf('  -> 已保存 JPG 预览: %s\n', current_jpg_path);
            else
                warning('  -> 未找到预期的 .raw 文件: %s', raw_file_name);
            end
        end
    end
end

