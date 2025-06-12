% % 一个自动化的数据准备管道脚本。
% 调用 create_R1_for_image_read 处理目录中的所有文件，将最终的 .raw 文件保存到 SourceToYang/ 文件夹中。它还保存 JPG 版本以便快速查看 。
% create_R1_for_image_read: 将由 MSTAR2JPG.m 生成的 .mat 文件转换为算法主程序期望的特定 .raw 二进制格式。它将所有幅度值连接在一起，然后是所有相位值 。
clear;
clc;
root = '.\step0_original2mat_SN_132'; % Specify the root path
files = dir(fullfile(root, '*.mat')); % List all .mat files
for i = 1:length(files)
    inputmat_file = fullfile(files(i).folder, files(i).name);
    output_basename = [files(i).name(1:end-4)]; % e.g., R1_T1 for T1.mat
    create_R1_for_image_read(inputmat_file, ['Step1_SourceToYang\' output_basename]);
    rawname = ['Step1_SourceToYang\' output_basename '.128x128.raw'];
    [fileimage, image_value] = image_read(rawname);
%     imagesc(fileimage);
    imwrite(uint8(imadjust(fileimage)*255),['Step2_JPGofSN_132\' output_basename '_v1.JPG']); % 调整对比度后保存
    imwrite(uint8(fileimage/max(max(fileimage))*255),['Step2_JPGofSN_132\' output_basename '_v2.JPG']); % 调整对比度后保存
end

