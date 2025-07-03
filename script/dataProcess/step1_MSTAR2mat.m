% =============================================================================
% step0_MSTAR2mat.m
% =============================================================================
%
% 功能:
% 该脚本自动处理整个MSTAR数据集。它会递归地遍历一个指定的源目录
% (例如 '00_Data_Raw')，将所有原始的MSTAR二进制数据文件转换为.mat格式，
% 并将它们保存在一个指定的目标目录 (例如 '01_Data_Processed') 中，
% 同时完整地保留原始的目录结构。
%
% 修正历史:
% - 此版本修正了先前版本中的一个严重错误，该错误导致在处理
%   头信息大于1024字节的文件时解析失败。
% - 恢复了逐行读取头信息的稳健方法，以确保无论头信息多长都能正确解析。
% - 保留了高效的递归目录处理和使用正则表达式进行键值对解析的功能。
%
% =============================================================================

clc;
clear;
close all;

% --- 配置 ---
% 获取当前脚本所在文件夹的路径，并由此确定项目根目录
try
    [scriptFolder, ~, ~] = fileparts(mfilename('fullpath'));
    % 从脚本文件夹返回到项目根目录
    projectRoot = fullfile(scriptFolder, '..');
catch
    % 如果在实时编辑器或无文件上下文的环境中运行，则使用当前工作目录
    projectRoot = pwd;
    warning('无法使用 mfilename 确定路径，将使用当前工作目录作为项目根目录。');
end

% 设置源路径和目标路径
sourceRoot = fullfile(projectRoot, 'datasets', 'SAR_ASC_Project', '00_Data_Raw');
targetRoot = fullfile(projectRoot, 'datasets', 'SAR_ASC_Project', '01_Data_Processed_mat');

% --- 主执行逻辑 ---
fprintf('开始处理...\n');
fprintf('源目录: %s\n', sourceRoot);
fprintf('目标目录: %s\n', targetRoot);

% 检查源目录是否存在
if ~isfolder(sourceRoot)
    error('源目录不存在或不是一个文件夹: %s', sourceRoot);
end

% 创建目标根目录（如果它不存在）
if ~isfolder(targetRoot)
    mkdir(targetRoot);
end

% 调用递归函数开始处理
process_directory(sourceRoot, targetRoot);

fprintf('所有文件处理完成。\n');


% --- 辅助函数 ---

function process_directory(currentSourcePath, currentTargetPath)
    % 获取当前源路径下的所有内容
    items = dir(currentSourcePath);
    
    for i = 1:length(items)
        itemName = items(i).name;
        
        % 忽略 '.' 和 '..' 目录以及macOS的 .DS_Store 等隐藏文件
        if strcmp(itemName, '.') || strcmp(itemName, '..') || startsWith(itemName, '.')
            continue;
        end
        
        sourceItemPath = fullfile(currentSourcePath, itemName);
        targetItemPath = fullfile(currentTargetPath, itemName);

        if items(i).isdir
            % 如果是目录，则创建对应的目标目录并进行递归
            if ~isfolder(targetItemPath)
                fprintf('创建目录: %s\n', targetItemPath);
                mkdir(targetItemPath);
            end
            process_directory(sourceItemPath, targetItemPath);
        else
            % 如果是文件，则进行处理
            fprintf('正在处理: %s\n', sourceItemPath);
            
            % 为输出文件构建.mat路径
            targetMatPath = [targetItemPath, '.mat'];
            if isfile(targetMatPath)
                fprintf('  -> 跳过，目标文件已存在: %s\n', targetMatPath);
                continue;
            end

            % --- 文件转换逻辑 (修正版) ---
            FID = -1; % 初始化文件ID
            try
                FID = fopen(sourceItemPath, 'rb', 'ieee-be');
                if FID == -1
                    warning('无法打开文件: %s', sourceItemPath);
                    continue;
                end
                
                % 初始化变量以防文件头不完整
                ImgColumns = [];
                ImgRows = [];
                TargetAz = [];
                headerEnded = false;

                % 逐行读取文件头，确保稳健性
                while ~feof(FID)
                    lineText = fgetl(FID);
                    if lineText == -1 % 文件结束或读取错误
                        break;
                    end
                    
                    if contains(lineText, 'NumberOfColumns')
                        tokens = regexp(lineText, '=\s*(\d+)', 'tokens', 'once');
                        if ~isempty(tokens), ImgColumns = str2double(tokens{1}); end
                    elseif contains(lineText, 'NumberOfRows')
                        tokens = regexp(lineText, '=\s*(\d+)', 'tokens', 'once');
                        if ~isempty(tokens), ImgRows = str2double(tokens{1}); end
                    elseif contains(lineText, 'TargetAz')
                        tokens = regexp(lineText, '=\s*([\d.-]+)', 'tokens', 'once'); % 允许负号和点
                        if ~isempty(tokens), TargetAz = str2double(tokens{1}); end
                    elseif contains(lineText, '[EndofPhoenixHeader]')
                        headerEnded = true;
                        break; % 找到头尾，退出循环
                    end
                end

                % 验证是否已读取所有元数据
                if isempty(ImgColumns) || isempty(ImgRows) || isempty(TargetAz) || ~headerEnded
                    warning('警告: 在 %s 中未能找到完整的元数据头。跳过此文件。', sourceItemPath);
                    fclose(FID);
                    continue;
                end

                % 文件指针现在正好在头信息之后，可以直接读取二进制数据
                Mag = fread(FID, 2 * ImgColumns * ImgRows, 'float32=>float32', 'ieee-be');
                fclose(FID); % 读取后关闭文件
                
                % 验证读取的数据量是否正确
                if length(Mag) < 2 * ImgColumns * ImgRows
                    warning('警告: %s 中的数据大小不足。跳过此文件。', sourceItemPath);
                    continue;
                end

                % 重塑数据为幅度和相位矩阵
                Img = reshape(Mag(1:(ImgColumns * ImgRows)), [ImgColumns, ImgRows]);
                phase = reshape(Mag((ImgColumns * ImgRows + 1):end), [ImgColumns, ImgRows]);

                % 保存到.mat文件
                save(targetMatPath, 'Img', 'phase', 'TargetAz');
                fprintf('  -> 已保存到: %s\n', targetMatPath);

            catch ME
                fprintf(2, '处理文件 %s 时发生错误: %s\n', sourceItemPath, ME.message);
                if FID ~= -1
                    fclose(FID);
                end
            end
            % --- 转换逻辑结束 ---
        end
    end
end
