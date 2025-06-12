% % 主算法使用的函数来读取自定义 .raw 文件。解析文件名以获取维度，读取二进制数据，并重构复值 SAR 图像（结合幅度和相位） 。
%读取数据文件中的数据信息
%image_file为要读取的数据文件名称(格式固定) 例如:'hb03333.015.128x128.raw'

function [fileimage image_value] = image_read(image_file)
image_filename_inver = fliplr(image_file);
index_dot = findstr(fliplr(image_file),'.');
if (size(index_dot) < 2)
    disp('file can not open')
    return;
end
image_Nsize = fliplr(image_filename_inver((index_dot(1) + 1):(index_dot(2) - 1)));
index_multi = findstr(image_Nsize, 'x');
image_width = str2num(image_Nsize(1: (index_multi(1) - 1)));
image_heigh =  str2num(image_Nsize((index_multi(1) + 1):end));

%读取数据文件 幅度magnitude  相位phase
[f_id,message] = fopen(image_file, 'r', 's');%注意打开时的文件方式ieee格式。
if(f_id == -1)
    errordlg(message)
    return;
end
image = linspace(0,0,image_width*image_heigh*2);
[image, count_cell] = fread(f_id,image_width*image_heigh*2,'float');
fclose(f_id);
magnitude =  image(1:count_cell/2) ;%强度,长度为count_cell/2。
%magnitude = reshape(magnitude,image_width,image_heigh );
%imshow(magnitude)
phase = image(count_cell/2 + 1 : end);%相位,长度为count_cell/2。

%图像显示
%min_cell = min(magnitude)
%max_cell = max(magnitude)
%b = magnitude(1:10)
real = zeros(1,image_width*image_heigh);% 实部real
imag = zeros(1,image_width*image_heigh);% 虚部 imag
image_magnitude = zeros(1,image_width*image_heigh);%强度
image_value = zeros(1,image_width*image_heigh);%像素值 存放复数.
for i = 1:(image_width*image_heigh);%循环中得到每个像素值
    temp_mag = magnitude(i);
    temp_phs = phase(i);
    real = temp_mag * cos( phase(i) );
    imag = temp_mag*sin( phase(i) );
%image_magnitude(i)  = log10(real*real + imag*imag);
 image_magnitude(i)  = sqrt(real*real + imag*imag);
 image_value(i) = real + sqrt(-1)* imag;
end
%image_magnitude, image_value, image_head
fileimage = reshape(image_magnitude, image_width, image_heigh);%图像强度 fileimage 128*128
%fileimage = reshape(log(image_magnitude + 0.001), image_width, image_heigh);
image_value = reshape(image_value, image_width, image_heigh);%复数图像 image_value 128*128




