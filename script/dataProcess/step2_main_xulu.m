%**xulu整理
% 从SourceToYang，文件夹中批量读取【raw】数据，提取的属性散射中心保存在‘OutputofYang’文件夹，根据属性散射中心重建的影像切片保存在‘ReconstructYang’文件夹
% 遍历所有准备好的 .raw 文件，调用核心提取函数( extrac )，并保存参数和重建结果 。
clear;
clc;
close all;

root = 'E:\Document\_Mission\2025\250512_三维目标散射机理研究与特征提取研究\MSTAR数据集\算法1_杨老师\datasourceProcess\Step1_SourceToYang\'; % Specify the root path
files = dir(fullfile(root, '*.raw')); % List all .raw files
outpath = 'E:\Document\_Mission\2025\250512_三维目标散射机理研究与特征提取研究\MSTAR数据集\算法1_杨老师\datasourceProcess\Step2_OutputofYang\';
comparepath = 'E:\Document\_Mission\2025\250512_三维目标散射机理研究与特征提取研究\MSTAR数据集\算法1_杨老师\datasourceProcess\Step2_ReconstructYang\';
for i = 1:length(files)
    inputraw_file = fullfile(files(i).folder, files(i).name);
    [fileimage, image_value] = image_read(inputraw_file);
    K=fileimage;
    K_complex=image_value;
    
    % % % %算法1主函数：
    scatter_all=extrac(K,K_complex);
    output_basename = [files(i).name(1:end-12) '_yang.mat'];
    save([outpath output_basename],'scatter_all');%保存提取的属性散射中心&参数
        
    % % % %算法1结果重建：
    % s=simulation(scatter_all);%图1：重建图
    % diff = fileimage-s;
    % reconstru_basename = [files(i).name(1:end-12) '_yangRecon.mat'];
    % full_path = fullfile(comparepath, reconstru_basename);
    % save(full_path,'s','diff');%保存重建的切片
    % fclose all;
    % %出图用，不保存，可以直接打开
    % gui_s=max(max(s));
    % fileimage= TargetDetect(fileimage,30 );
    % gui_f=max(max(fileimage));
    % s_gui=s/gui_s;
    % fileimage_gui=fileimage/gui_f;
    % C = normxcorr2(s_gui,fileimage_gui);
    % max(max(C))
    % figure,imshow(C);
    % colormap(jet);colorbar;%图2
    % figure,imshow(fileimage);%图3：原图
    
end

