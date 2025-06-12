% % 恒假警报率（CFAR）算法。任务是创建一个二进制掩码，将目标像素与背景杂波分离，确保后续步骤仅关注目标本身 。
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
%第二组阈值分割
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
% 目标抑制
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