% TargetDetect.m — CFAR-based target detection for SAR tiles
%
% 概述:
% - 输入: Img (二维实数强度图), winsize (局部窗口边长)
% - 输出: DetRS (与 Img 同尺寸; 目标区域保留原像素强度, 其他位置为 0)
%
% 方法流程概述:
% 1) 背景统计与全局 CFAR 阈值
%    - 仅用图像四个边框区域像素估计背景均值 ave 和标准差 var
%    - 依据虚警率 PF1 推导阈值系数 tao, 得到阈值: thd1 = tao*var + ave
% 2) 初检 (全局阈值)
%    - 将 Img >= thd1 的像素标为候选; 为避免越界, 边界处 (winsize 宽度内) 清零
% 3) 局部纹理/方差抑制
%    - 对每个候选点, 取中心为该点、边长为 winsize 的局部块, 计算标准差 var1
%    - 若 var1 < MULTI * var (MULTI 为超参数), 视为背景高亮而剔除
% 4) 空间聚类与形态学后处理
%    - 用 8 邻域核 [1 1 1;1 0 1;1 1 1] 做 conv2 统计邻居数, 仅保留邻居数>3 的像素
%    - 使用半径为 4 的圆盘结构元素进行闭运算 (imclose) 平滑边界、填补小孔洞
% 5) 强度掩膜输出
%    - 最终二值检测图与原始 Img 相乘, 输出 DetRS 为仅含目标区域强度的图
%
% 主要超参数:
% - PF1  : 全局 CFAR 虚警率 (越小阈值越高, 误警越低)
% - MULTI: 局部标准差门限的倍率系数 (抑制低纹理伪警)
% - winsize: 局部窗口大小 (代码以 winsize/2 为半径处理边界)
%
% 复杂度与实现注意:
% - 主时耗来自局部方差的两重循环与一次 conv2, 复杂度约 O(row*col)
% - 可用向量化/滑动窗口统计或 blockproc 替代逐点循环以提速
% - conv2 可与 gpuArray 协同以做 GPU 加速; 形态学函数需工具箱支持
% - 输出为强度掩膜而非纯二值图 (便于后续度量或可视化)
%
% 使用示例:
%   DetRS = TargetDetect(Img, 30);
%
% 依赖函数/工具箱:
% - conv2, strel, imclose (Image Processing Toolbox)

%detect targets from MStar tiles
function [ DetRS ] = TargetDetect( Img,winsize )
localsize=winsize/2;
%Img= medfilt2(Img1);
[row,col]=size(Img);
DetRs=zeros(row,col);
PF1=0.00001;
MULTI=2.9;
%get the segmentation threshold
k1=sqrt(3.1415/2);
k2=sqrt(2-3.1415/2);
tao=(sqrt(-2*log10(PF1))-k1)/k2;  %CFAR
%%%%%%%%%%%%%%%%%%%%%%%%%
left=Img(:,1:winsize);
right=Img(:,(col-winsize+1):col);
top=Img(1:winsize,:);
bottom=Img((row-winsize+1):row,:);
temp=[left(:);right(:);top(:);bottom(:)];
ave=mean(temp);
var=std(temp);
thd1=tao*var+ave;
temp1=DetRs(:);
temp1(find(Img>=thd1))=1;
DetRS=reshape(temp1,row,col);%(winsize+1):(row-winsize),(winsize+1):(col-winsize)
DetRS(:,1:winsize)=0;
DetRS(:,(col-winsize+1):col)=0;
DetRS(1:winsize,:)=0;
DetRS((row-winsize+1):row,:)=0;
%�ڶ�����ֵ�ָ�
for i=1:row
    for j=1:col
        if DetRS(i,j)>0
            l=max((i-localsize),1);
            r=min((i+localsize),row);
            t=max(1,(j-localsize));
            b=min(col,(j+localsize));
            tt=Img(l:r,t:b);
            tt=tt(:);
            var1=std(tt);
            if var1<MULTI*var
             DetRS(i,j)=0;
            end
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ŀ������
% figure;imshow(DetRS);%colormap;colorbar;
filter=[1,1,1;1,0,1;1,1,1];
filtered=conv2(DetRS,filter);
DetRS=filtered(2:(row+1),2:(col+1)).*DetRS;
temp=DetRS(:);
temp(find(temp<=3))=0;
temp(find(temp>0))=1;
DetRS=reshape(temp,row,col);
se = strel('disk',4);
DetRS=imclose(DetRS,se);

DetRS=Img.*DetRS;
%imwrite(DetRS,)
%figure;imshow(DetRS);%colormap;colorbar