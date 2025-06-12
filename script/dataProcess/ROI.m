function image_interest=ROI(matrix_image,R);             %%%%%%%%%%%%%matrix_image表示用分水岭分割得到的模板，R表示该模板中某个感兴趣区域的标记
%%%%%%%%%%%%%%%该函数中就是将模板中特定标记的感兴趣区域提取出来,然后将该模板中的标记都标为1，其他标记都为0%%%%%%%%%%%55



image_interest=matrix_image-matrix_image;
  

[x,y]=find(matrix_image==R);                        %%%%%%%%%%%找到某个标记下的区域像素点的坐标位置的集合
% [x,y]=find(matrix_image~=0); 
[width,height]=size(x);                         %%%%%%%%%%%x,y的矩阵大小是一样的，所以求得x的size可以用来代替y的。height等于1



for i=1:width
    image_interest(x(i,1),y(i,1))=1;
end

    