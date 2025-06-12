% % watershed_image 实现了分水岭分割算法。
% 它将目标区域进一步划分为更小的区域（感兴趣区域，或 ROIs），每个区域理想情况下仅包含一个散射中心。这类似于光学图像中的实例分割 。

%function  segmentation=watershed_image(magnitude)

function  [y1 y2 R1 R2]=watershed_image(magnitude)

max_cell = max(max(magnitude));
[max_i,max_j]=find(magnitude==max_cell);    %找到最大值以及其横纵坐标值

threshold_3db=max_cell/(10^(3/20));
% threshold_3db=max_cell/(3);
threshold_20db=max_cell/(3);

[i_3db,j_3db]=find(magnitude>=threshold_3db);                        %峰值以下3db的所有点集合：i_3db是这些点横坐标的集合，j_3db是这些点纵坐标的集合
% [i_20db,j_20db]=find((threshold_3db>=magnitude)&(magnitude>=threshold_20db));      %峰值以下3db至20db的所有点集合
[i_20db,j_20db]=find((threshold_20db<=magnitude)); 

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


for j=1:width_i_3db-1
    for i=1:width_i_3db-j
        if magnitude(i_3db(i,1),j_3db(i,1))<=magnitude(i_3db(i+1,1),j_3db(i+1,1))
            i_tempt=i_3db(i,1);
            j_tempt=j_3db(i,1);
            i_3db(i,1)=i_3db(i+1,1);
            j_3db(i,1)=j_3db(i+1,1);
            i_3db(i+1,1)=i_tempt;
            j_3db(i+1,1)=j_tempt;
        end
    end
end                                              %%此时得到的峰值以下3db的所有点集合是按照从大到小的顺序排列的

for j=1:width_i_20db-1
    for i=1:width_i_20db-j
        if magnitude(i_20db(i,1),j_20db(i,1))<=magnitude(i_20db(i+1,1),j_20db(i+1,1))
            i_tempt=i_20db(i,1);
            j_tempt=j_20db(i,1);
            i_20db(i,1)=i_20db(i+1,1);
            j_20db(i,1)=j_20db(i+1,1);
            i_20db(i+1,1)=i_tempt;
            j_20db(i+1,1)=j_tempt;
        end
    end
end                                              %%此时得到的峰值以下3db至20db的所有点集合是按照从大到小的顺序排列的


plate=zeros(3,3);
R1=1;                              %R标记同一分割区域
for k=1:width_i_3db
    temp=0;
    if i_3db(k,1)~=1 & j_3db(k,1)~=1 & i_3db(k,1)~=128 & j_3db(k,1)~=128
    plate(1,1)=y1(i_3db(k,1)-1,j_3db(k,1)-1);
    plate(1,2)=y1(i_3db(k,1)-1,j_3db(k,1));
    plate(1,3)=y1(i_3db(k,1)-1,j_3db(k,1)+1);
    plate(2,1)=y1(i_3db(k,1),j_3db(k,1)-1);
    plate(2,2)=y1(i_3db(k,1),j_3db(k,1));
    plate(2,3)=y1(i_3db(k,1),j_3db(k,1)+1);
    plate(3,1)=y1(i_3db(k,1)+1,j_3db(k,1)-1);
    plate(3,2)=y1(i_3db(k,1)+1,j_3db(k,1));
    plate(3,3)=y1(i_3db(k,1)+1,j_3db(k,1)+1);
    
    max_plate=max(max(plate));
    if max_plate<=1
        R1=R1+1;
        y1(i_3db(k,1),j_3db(k,1))=R1;
    else
        temp=max_plate;
        for i=1:3
            for j=1:3
                if plate(i,j)<=temp & plate(i,j)>1
                    temp=plate(i,j);
                    plate(i,j)=-1;
                end
            end
        end
         y1(i_3db(k,1),j_3db(k,1))=temp;
         for i=1:3
            for j=1:3
                if plate(i,j)==-1
                    y1(i_3db(k,1)-2+i,j_3db(k,1)-2+j)=temp;
                end
            end
        end
    end
    end
end                                                                    %%%%%对峰值以下3db的所有点进行分割

             %%%%%%%%%%%%%%%%R1是初始时划分给最大峰值下3db点的区域的数字数目，但不一定是区域的个数，%%%%%%%%%%
                       %%%%%%%%%%%%%%%%%%%因为可能在划分区域的时候标记一定数字的某些区域会被合并，造成某一数
                       %%%%%%%%%%%%%%%%%%%字下的区域变成空集，这在后面的优化中会解决。                     %%%%%%%%%%%
            

plate=zeros(3,3);   
R2=1;
for k=1:width_i_20db
    temp=0;
    if i_20db(k,1)~=1 & j_20db(k,1)~=1 & i_20db(k,1)~=128 & j_20db(k,1)~=128
    plate(1,1)=y2(i_20db(k,1)-1,j_20db(k,1)-1);
    plate(1,2)=y2(i_20db(k,1)-1,j_20db(k,1));
    plate(1,3)=y2(i_20db(k,1)-1,j_20db(k,1)+1);
    plate(2,1)=y2(i_20db(k,1),j_20db(k,1)-1);
    plate(2,2)=y2(i_20db(k,1),j_20db(k,1));
    plate(2,3)=y2(i_20db(k,1),j_20db(k,1)+1);
    plate(3,1)=y2(i_20db(k,1)+1,j_20db(k,1)-1);
    plate(3,2)=y2(i_20db(k,1)+1,j_20db(k,1));
    plate(3,3)=y2(i_20db(k,1)+1,j_20db(k,1)+1);
    
    max_plate=max(max(plate));
    if max_plate<=1
        R2=R2+1;
        y2(i_20db(k,1),j_20db(k,1))=R2;
    else
        temp=max_plate;
        for i=1:3
            for j=1:3
                if plate(i,j)<=temp & plate(i,j)>1
                    temp=plate(i,j);
                    plate(i,j)=-1;
                end
            end
        end
         y2(i_20db(k,1),j_20db(k,1))=temp;
         for i=1:3
            for j=1:3
                if plate(i,j)==-1
                    y2(i_20db(k,1)-2+i,j_20db(k,1)-2+j)=temp;
                end
            end
        end
    end
    end
end                   
 
y3=y1+y2;                                      %%%%%%y3就是所有从最大峰值点到最大峰值下20db的所有标记分割区域
for i=1:magnitude_i
    for j=1:magnitude_j
        if y1(i,j)~=0
            y1(i,j)=y1(i,j)-1;
        end
        if y2(i,j)~=0
            y2(i,j)=y2(i,j)-1;
        end
    end
end                                          %%%%%%%%%%%%%%%%这个循环的目的仅仅是为了让标记区域从1开始标记，美化作用  


for k=1:width_i_3db
    temp=0;
    if i_3db(k,1)~=1 & j_3db(k,1)~=1 & i_3db(k,1)~=128 & j_3db(k,1)~=128
       plate(1,1)=y1(i_3db(k,1)-1,j_3db(k,1)-1);
       plate(1,2)=y1(i_3db(k,1)-1,j_3db(k,1));
       plate(1,3)=y1(i_3db(k,1)-1,j_3db(k,1)+1);
       plate(2,1)=y1(i_3db(k,1),j_3db(k,1)-1);
       plate(2,2)=0;
       plate(2,3)=y1(i_3db(k,1),j_3db(k,1)+1);
       plate(3,1)=y1(i_3db(k,1)+1,j_3db(k,1)-1);
       plate(3,2)=y1(i_3db(k,1)+1,j_3db(k,1));
       plate(3,3)=y1(i_3db(k,1)+1,j_3db(k,1)+1);
         
          n=0;
          for i=1:3
              for j=1:3
                  if plate(i,j)==y1(i_3db(k,1),j_3db(k,1));
                     n=n+1;                                 %%%%%%n是用来表示在该区域中和y1(i_3db(k,1),j_3db(k,1))数值相等的像素点的个数
                  end
              end
          end
          if n==0                                            %%%%%%n==0表示没有和y1(i_3db(k,1),j_3db(k,1))数值相等的像素点
              max_plate=max(max(plate));
              if max_plate==0                             %%%%%%%%%%这种情况说明y1(i_3db(k,1),j_3db(k,1))是个孤立的点，邻域其它8个点都为0；
                 y1(i_3db(k,1),j_3db(k,1))=0;
              else                                         %%%%%%%%这种情况说明y1(i_3db(k,1),j_3db(k,1))被割开了，此时需把%%%%%%%%%
                                                          %%%%%%%y1(i_3db(k,1),j_3db(k,1))并入其邻域的某个区域中                                               
                  temp=max_plate;
                  for i=1:3
                      for j=1:3
                          if plate(i,j)<=temp & plate(i,j)~=0
                             temp=plate(i,j);
                          end
                      end
                  end
                  y1(i_3db(k,1),j_3db(k,1))=temp;
              end
          end
    end 
end                                                     %%%%%%%%%%%%%%%%%%%%%%%对y1模板进行优化
     


for k=1:width_i_20db
    temp=0;
    if i_20db(k,1)~=1 & j_20db(k,1)~=1 & i_20db(k,1)~=128 & j_20db(k,1)~=128
         plate(1,1)=y2(i_20db(k,1)-1,j_20db(k,1)-1);
         plate(1,2)=y2(i_20db(k,1)-1,j_20db(k,1));
         plate(1,3)=y2(i_20db(k,1)-1,j_20db(k,1)+1);
         plate(2,1)=y2(i_20db(k,1),j_20db(k,1)-1);
         plate(2,2)=0;
         plate(2,3)=y2(i_20db(k,1),j_20db(k,1)+1);
         plate(3,1)=y2(i_20db(k,1)+1,j_20db(k,1)-1);
         plate(3,2)=y2(i_20db(k,1)+1,j_20db(k,1));
         plate(3,3)=y2(i_20db(k,1)+1,j_20db(k,1)+1);
         
          n=0;
          for i=1:3
              for j=1:3
                  if plate(i,j)==y2(i_20db(k,1),j_20db(k,1));
                     n=n+1;                                 %%%%%%n是用来表示在该区域中和y2(i_20db(k,1),j_20db(k,1))数值相等的像素点的个数
                  end
              end
          end
          if n==0                                            %%%%%%n==0表示没有和y2(i_20db(k,1),j_20db(k,1))数值相等的像素点
              max_plate=max(max(plate));
              if max_plate==0                             %%%%%%%%%%这种情况说明y2(i_20db(k,1),j_20db(k,1))是个孤立的点，邻域其它8个点都为0；
%                  y2(i_20db(k,1),j_20db(k,1))=0;
              else                                         %%%%%%%%这种情况说明y2(i_20db(k,1),j_20db(k,1))被割开了，此时需把%%%%%%%%%
                                                          %%%%%%%y2(i_20db(k,1),j_20db(k,1))并入其邻域的某个区域中                                               
                  temp=max_plate;
                  for i=1:3
                      for j=1:3
                          if plate(i,j)<=temp & plate(i,j)~=0
                             temp=plate(i,j);
                          end
                      end
                  end
                  y2(i_20db(k,1),j_20db(k,1))=temp;
              end
          end
    end 
end
                                                       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%对y2模板进行优化

%%%%%%%%%%%%%%%%%%%%%%%%下面针对R1、R2在递增的过程中可能由于区域合并使得某些数值会出现空集的情况做优化
R1=R1-1;
R2=R2-1;
T1=R1;
T2=R2;
y1_copy=y1;
y2_copy=y2;

for i=1:R1
    [x_R1,y_R1]=find(y1==i);
    if isempty(x_R1)
      T1=T1-1;
        for k=1:width_i_3db
           if y1(i_3db(k,1),j_3db(k,1))>i
               y1_copy(i_3db(k,1),j_3db(k,1))=y1_copy(i_3db(k,1),j_3db(k,1))-1;
           end
        end
    end
end
y1=y1_copy;
R1=T1;

for i=1:R2
    [x_R2,y_R2]=find(y2==i);
    if isempty(x_R2)
        T2=T2-1;
        for k=1:width_i_20db
           if y2(i_20db(k,1),j_20db(k,1))>i
              y2_copy(i_20db(k,1),j_20db(k,1))=y2_copy(i_20db(k,1),j_20db(k,1))-1;
           end
        end
    end
end
y2=y2_copy;
R2=T2;


%%%%%%%%%%优化该算法，合并某些区域%%%%%%%%%
%%%%%%%%%%合并的准则是：按顺序找到图像矩阵中两个相连标记的区域，如果这两个相邻标记区域相连，则把它们作为一个标记区域，否则查找下两个个相连标记
%%%%%%%%%%区域
for i=1:R1-1
    segment1=ROI(y1,i);
    segment2=ROI(y1,i+1);
    segment=segment1+segment2;
    [L,num]=bwlabel(segment,8);
    
    if num==1
       [x_R1,y_R1]=find(y1==i);
       [x_size,temp]=size(x_R1);
       for j=1:x_size
           y1(x_R1(j),y_R1(j))=i+1;
       end
    end
end

for i=1:R2-1
    segment1=ROI(y2,i);
    segment2=ROI(y2,i+1);
    segment=segment1+segment2;
    [L,num]=bwlabel(segment,8);
    
    if num==1
       [x_R2,y_R2]=find(y2==i);
       [x_size,temp]=size(x_R2);
       for j=1:x_size
           y2(x_R2(j),y_R2(j))=i+1;
       end
    end
end


T1=R1;
T2=R2;
y1_copy=y1;
y2_copy=y2;

for i=1:R1
    [x_R1,y_R1]=find(y1==i);
    if isempty(x_R1)
      T1=T1-1;
        for k=1:width_i_3db
           if y1(i_3db(k,1),j_3db(k,1))>i
               y1_copy(i_3db(k,1),j_3db(k,1))=y1_copy(i_3db(k,1),j_3db(k,1))-1;
           end
        end
    end
end
y1=y1_copy;
R1=T1;

for i=1:R2
    [x_R2,y_R2]=find(y2==i);
    if isempty(x_R2)
        T2=T2-1;
        for k=1:width_i_20db
           if y2(i_20db(k,1),j_20db(k,1))>i
              y2_copy(i_20db(k,1),j_20db(k,1))=y2_copy(i_20db(k,1),j_20db(k,1))-1;
           end
        end
    end
end
y2=y2_copy;
R2=T2;

    
  
    
    
               
               
               