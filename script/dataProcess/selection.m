% % 模型确定。
% 对于watershed_image中的每个 ROI，该函数决定其中的散射中心是局部化（点状）还是分布化（线状）。它通过分析 ROI 的惯性矩和局部峰的数量来做出这一决定。

function  [type,temp_coordinate]=selection(segmented_image);

%%%%%%%%确定散射中心类型和模型阶次%%%%%%%%%%%%%%
%%%%%%%%输入是一幅ROI后的图像，即只有标记为特定数值的图像位置有幅度值，其它位置幅度值为零；输出的是散射中心的类型和模型阶次%%%%%%%

[image_x,image_y]=size(segmented_image);

[x,y]=find(segmented_image~=0);   %%%%%找到模版中不为零的那些坐标位置

size_thresh=2;
size_x=size(x);
size_segmented=size_x(1,1);
if size_segmented<=size_thresh
   type=-1;
   temp_coordinate=[];
else

x_min=min(x);                     %%%%%%%找到坐标中的那些极值
x_max=max(x);
y_min=min(y);
y_max=max(y);

[x_size,temp]=size(x);           %%%%%%%%%%x_size是不为零的数的个数，temp等于1

%%%%%%%%%%%%%%%下面就是套用公式的过程%%%%%%%%%%%%%%%%%%
Cx_temp=0;                      
Cy_temp=0;
C_temp=0;

for i=1:x_size
    x_v=x(i)-x_min+1;
    y_v=y(i)-y_min+1;
    Cx_temp=x_v*segmented_image(x(i),y(i))+Cx_temp;
    Cy_temp=y_v*segmented_image(x(i),y(i))+Cy_temp;
    C_temp=segmented_image(x(i),y(i))+C_temp;
end

Cx=Cx_temp/C_temp;
Cy=Cy_temp/C_temp;

I_x=0;
I_y=0;
for i=1:x_size
    x_v=x(i)-x_min+1;
    y_v=y(i)-y_min+1;
    I_x=((x_v-Cx)^2)*segmented_image(x(i),y(i))+I_x;
    I_y=((y_v-Cy)^2)*segmented_image(x(i),y(i))+I_y;
end 
k=I_x/I_y;
% mesh(segmented_image);

%%%%%%%%%%%%%%%%%%%下面根据所得到的k值来判断该区域中包含的是分布散射中心还是局部散射中心，如果是局部散射中心则进一步判断该区域中局
%%%%%%%%%%%%%%%%%%%部散射中心的个数，另外还要确定分布散射中心和局部散射中心的坐标位置。

temp_coordinate=zeros(x_size,2);
s=1;
number=0;         




   plate=zeros(3,3);
   for i=1:x_size
       plate(1,1)= segmented_image(x(i)-1,y(i)-1);
       plate(1,2)= segmented_image(x(i)-1,y(i));
       plate(1,3)= segmented_image(x(i)-1,y(i)+1);
       plate(2,1)= segmented_image(x(i),y(i)-1);
       plate(2,2)= segmented_image(x(i),y(i));
       plate(2,3)= segmented_image(x(i),y(i)+1);
       plate(3,1)= segmented_image(x(i)+1,y(i)-1);
       plate(3,2)= segmented_image(x(i)+1,y(i));
       plate(3,3)= segmented_image(x(i)+1,y(i)+1);
    
       if plate(2,2)==max(max(plate))
           number=number+1;
           temp_coordinate(s,1)=x(i);
           temp_coordinate(s,2)=y(i);
           s=s+1;
       end
        
   end



 if number<=1
     if k<3
        type=1;
     end
     
     if k>=3
        type=0;
     end
     
     temp_coordinate(1,1)=Cx+x_min-1;
     temp_coordinate(1,2)=Cy+y_min-1;  
 end
 
 if number==2
     if k<=1.5
         type=2;
     end
     if k>1.5
         if abs(temp_coordinate(1,2)-temp_coordinate(2,2))<2
             type=0;
             temp_coordinate=zeros(1,2);
             temp_coordinate(1,1)=Cx+x_min-1;
             temp_coordinate(1,2)=Cy+y_min-1;  
         else
             type=2;
         end
     end
 end
 
if number>2
   type=number; 
end                                                          %%%%%%%%%%%%%%当type为0时，表示分布散射中心；
                                                             %%%%%%%%%%%%%%当type为1时，表示局部散射中心;
      type_order=type;                                                          %%%%%%%%%%%%%%当type大于1时，寻找该区域中的最强散射中心先处理


if type>1
    
   
   if k>1.5
      max_pix=max(max(segmented_image));
      [max_coor(1),max_coor(2)]=find(max_pix==segmented_image);
      for i=1:type_order
          if (temp_coordinate(i,1)~=max_coor(1,1)) | (temp_coordinate(i,1)~=max_coor(1,2)) 
              if abs(max_coor(1,2)-temp_coordinate(2,2))<2
                  type=0;
                  
                  temp_image=zeros(image_x,image_y);
                  
                  temp_xmin=min(max_coor(1,1),temp_coordinate(2,1));
                  temp_xmax=max(max_coor(1,1),temp_coordinate(2,1));
                  temp_ymin=min(max_coor(1,2),temp_coordinate(2,2));
                  temp_ymax=max(max_coor(1,2),temp_coordinate(2,2));
                  
                  temp_image((temp_xmin-1):(temp_xmax+1),(temp_ymin-1):(temp_ymax+1))=1;
                  segmented_image=segmented_image.*temp_image;           %%%%%%仅保留分布散射中心那一区域
                  [x,y]=find(segmented_image~=0);                        %%%%%找到模版中不为零的那些坐标位置

                  size_x=size(x);
                  size_segmented=size_x(1,1);
                  if size_segmented<=size_thresh
                     type=-1;
                  else

                  x_min=min(x);                                         %%%%%%%找到坐标中的那些极值
                  x_max=max(x);
                  y_min=min(y);
                  y_max=max(y);

                  [x_size,temp]=size(x);                                %%%%%%%%%%x_size是不为零的数的个数，temp等于1

                                                                        %%%%%%%%%%%%%%%下面就是套用公式的过程%%%%%%%%%%%%%%%%%%
                  Cx_temp=0;                      
                  Cy_temp=0;
                  C_temp=0;

                  for i=1:x_size
                      x_v=x(i)-x_min+1;
                      y_v=y(i)-y_min+1;
                      Cx_temp=x_v*segmented_image(x(i),y(i))+Cx_temp;
                      Cy_temp=y_v*segmented_image(x(i),y(i))+Cy_temp;
                      C_temp=segmented_image(x(i),y(i))+C_temp;
                  end

                  Cx=Cx_temp/C_temp;
                  Cy=Cy_temp/C_temp;
                 
                  temp_coordinate=zeros(1,2);
                  temp_coordinate(1,1)=Cx+x_min-1;
                  temp_coordinate(1,2)=Cy+y_min-1;  
                  break;
                  end
              end
          end
      end
                  
                  
                  if type>1
                      sum=[0,0];
                      
       
                      for i=1:type
                          average=temp_coordinate(i,:)+sum;
                          sum=average;
                      end
      
                      x_average=average(1,1)/type;
                      y_average=average(1,2)/type;
      
                      cor_now=temp_coordinate(1,:);
                      piancha=abs(x_average-temp_coordinate(1,1));
      
                      for i=2:type
                          temp_piancha=abs(x_average-temp_coordinate(i,1));
                          if temp_piancha>piancha
                             cor_now=temp_coordinate(i,:);
                          end
                      end                                                            %%%%%%%%%%%%%%%%找到所有峰值像素距离向位置与平均值偏差最大的那个峰值，作为局部散射中心处理
      
                      temp_coordinate=zeros(1,1);
                      temp_coordinate=cor_now;
       
                      type=1;
                      temp_image=zeros(image_x,image_y);
                      
                      temp_xmin=(temp_coordinate(1,1)-1);
                      temp_xmax=(temp_coordinate(1,1)+1);
                      temp_ymin=(temp_coordinate(1,2)-1);
                      temp_ymax=(temp_coordinate(1,2)+1);
                      temp_image(temp_xmin:temp_xmax,temp_ymin:temp_ymax)=1;
                      
                      segmented_image=segmented_image.*temp_image;    %%%%%%%仅保留局部散射中心的那一部分
       
                      [x,y]=find(segmented_image~=0);                        %%%%%找到模版中不为零的那些坐标位置

                       size_x=size(x);
                       size_segmented=size_x(1,1);
                       if size_segmented<=size_thresh
                          type=-1;
                       else

                           x_min=min(x);                                         %%%%%%%找到坐标中的那些极值
                           x_max=max(x);
                           y_min=min(y);
                           y_max=max(y);

                           [x_size,temp]=size(x);                                %%%%%%%%%%x_size是不为零的数的个数，temp等于1

                                                                        %%%%%%%%%%%%%%%下面就是套用公式的过程%%%%%%%%%%%%%%%%%%
                           Cx_temp=0;                      
                           Cy_temp=0;
                           C_temp=0;

                           for i=1:x_size
                               x_v=x(i)-x_min+1;
                               y_v=y(i)-y_min+1;
                               Cx_temp=x_v*segmented_image(x(i),y(i))+Cx_temp;
                               Cy_temp=y_v*segmented_image(x(i),y(i))+Cy_temp;
                               C_temp=segmented_image(x(i),y(i))+C_temp;
                           end

                           Cx=Cx_temp/C_temp;
                           Cy=Cy_temp/C_temp;
                 
                          temp_coordinate=zeros(1,2);
                          temp_coordinate(1,1)=Cx+x_min-1;
                          temp_coordinate(1,2)=Cy+y_min-1;  
                   
                       end
                  end
             end




   if k<=1.5
      
       sum=[0,0];
       
      for i=1:type
           average=temp_coordinate(i,:)+sum;
           sum=average;
      end
      
      x_average=average(1,1)/type;
      y_average=average(1,2)/type;
      
      cor_now=temp_coordinate(1,:)
      piancha=abs(x_average-temp_coordinate(1,1));
      
      for i=2:type_order
          temp_piancha=abs(x_average-temp_coordinate(i,1));
          if temp_piancha>piancha
              cor_now=temp_coordinate(i,:);
          end
      end                                                            %%%%%%%%%%%%%%%%找到所有峰值像素距离向位置与平均值偏差最大的那个峰值，作为局部散射中心处理
      
       temp_coordinate=zeros(1,1)
       temp_coordinate=cor_now;
       
       type=1;
       temp_image=zeros(image_x,image_y);
       
       
       temp_xmin=(temp_coordinate(1,1)-1);
       temp_xmax=(temp_coordinate(1,1)+1);
       temp_ymin=(temp_coordinate(1,2)-1);
       temp_ymax=(temp_coordinate(1,2)+1);
       temp_image(temp_xmin:temp_xmax,temp_ymin:temp_ymax)=1;
       
       segmented_image=segmented_image.*temp_image;    %%%%%%%仅保留局部散射中心的那一部分
       
       [x,y]=find(segmented_image~=0);                        %%%%%找到模版中不为零的那些坐标位置

                  size_x=size(x);
                  size_segmented=size_x(1,1);
                  if size_segmented<=size_thresh
                     type=-1;
                  else

                      x_min=min(x);                                         %%%%%%%找到坐标中的那些极值
                      x_max=max(x);
                      y_min=min(y);
                      y_max=max(y);

                     [x_size,temp]=size(x);                                %%%%%%%%%%x_size是不为零的数的个数，temp等于1

                                                                        %%%%%%%%%%%%%%%下面就是套用公式的过程%%%%%%%%%%%%%%%%%%
                     Cx_temp=0;                      
                     Cy_temp=0;
                     C_temp=0;

                      for i=1:x_size
                          x_v=x(i)-x_min+1;
                          y_v=y(i)-y_min+1;
                          Cx_temp=x_v*segmented_image(x(i),y(i))+Cx_temp;
                          Cy_temp=y_v*segmented_image(x(i),y(i))+Cy_temp;
                          C_temp=segmented_image(x(i),y(i))+C_temp;
                      end

                      Cx=Cx_temp/C_temp;
                      Cy=Cy_temp/C_temp;
                 
                      temp_coordinate=zeros(1,2);
                      temp_coordinate(1,1)=Cx+x_min-1;
                      temp_coordinate(1,2)=Cy+y_min-1;  
                  end
        end
   end
end