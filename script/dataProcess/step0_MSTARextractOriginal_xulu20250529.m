clc;clear
sourcePath = 'E:\Document\_Mission\2025\250512_三维目标散射机理研究与特征提取研究\MSTAR数据集\1原始\MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY\TARGETS\TEST\15_DEG\T72\SN_132\'; 
targetPath = 'E:\Document\_Mission\2025\250512_三维目标散射机理研究与特征提取研究\MSTAR数据集\算法1_杨老师\datasourceProcess\step0_original2mat_SN_132';
fileList = dir(sourcePath);  % 获取文件夹下所有内容

% 过滤掉.和..文件夹
files = fileList(~ismember({fileList.name}, {'.', '..'}));

% 遍历所有文件
for i = 1:length(files)
    orgin = [ sourcePath files(i).name];
    target = [targetPath files(i).name '.mat'];   
    MSTAR2JPG(sourcePath, targetPath);
end
