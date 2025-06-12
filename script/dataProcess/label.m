function  [y1 y2 R1 R2]=label(magnitude)

max_cell = max(max(magnitude));
[max_i,max_j]=find(magnitude==max_cell);    %找到最大值以及其横纵坐标值

threshold_3db=max_cell/(10^(3/20));
threshold_20db=max_cell/(2);

[i_3db,j_3db]=find(magnitude>=threshold_3db);                        %峰值以下3db的所有点集合：i_3db是这些点横坐标的集合，j_3db是这些点纵坐标的集合
[i_20db,j_20db]=find((threshold_3db>=magnitude)&(magnitude>=threshold_20db));      %峰值以下3db至20db的所有点集合


[width_i_3db,height_i_3db]=size(i_3db);         %width_i_3db表示峰值以下3db的点的个数，height_i_3db=1
%[width_j_3db,height_j_3db]=size(j_3db);        %因为纵坐标和横坐标是一一对应的，因此此行命令可以省去%width_i_3db

[width_i_20db,height_i_20db]=size(i_20db);
%[width_j_20db,height_i_20db]=size(j_20db);           %同上
   

[magnitude_i,magnitude_j]=size(magnitude);    %计算magnitude矩阵(原矩阵)的大小

y1=zeros(magnitude_i,magnitude_j);             %%%%y1用来标记最大峰值到其下3db的所有像素点被分割的区域
y2=zeros(magnitude_i,magnitude_j);             %%%%y2用来标记最大峰值下3db到最大峰值下20db的所有像素点被分割的区域

for i=1:width_i_3db
    y1(i_3db(i,1),j_3db(i,1))=1;                  %获得3db幅度像素模版
end
for i=1:width_i_20db
    y2(i_20db(i,1),j_20db(i,1))=1;                 %获得20db幅度像素模版
end    
[y1,R1]=bwlabel(y1,8);
[y2,R2]=bwlabel(y2,8);

