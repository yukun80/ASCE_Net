% % watershed_image ʵ���˷�ˮ��ָ��㷨��
% ����Ŀ�������һ������Ϊ��С�����򣨸���Ȥ���򣬻� ROIs����ÿ��������������½�����һ��ɢ�����ġ��������ڹ�ѧͼ���е�ʵ���ָ� ��

%function  segmentation=watershed_image(magnitude)

function  [y1 y2 R1 R2]=watershed_image(magnitude)

max_cell = max(max(magnitude));
[max_i,max_j]=find(magnitude==max_cell);    %�ҵ����ֵ�Լ����������ֵ

threshold_3db=max_cell/(10^(3/20));
% threshold_3db=max_cell/(3);
threshold_20db=max_cell/(3);

[i_3db,j_3db]=find(magnitude>=threshold_3db);                        %��ֵ����3db�����е㼯�ϣ�i_3db����Щ�������ļ��ϣ�j_3db����Щ��������ļ���
% [i_20db,j_20db]=find((threshold_3db>=magnitude)&(magnitude>=threshold_20db));      %��ֵ����3db��20db�����е㼯��
[i_20db,j_20db]=find((threshold_20db<=magnitude)); 

[width_i_3db,height_i_3db]=size(i_3db);         %width_i_3db��ʾ��ֵ����3db�ĵ�ĸ�����height_i_3db=1
%[width_j_3db,height_j_3db]=size(j_3db);        %��Ϊ������ͺ�������һһ��Ӧ�ģ���˴����������ʡȥ%width_i_3db

[width_i_20db,height_i_20db]=size(i_20db);
%[width_j_20db,height_i_20db]=size(j_20db);           %ͬ��
   

[magnitude_i,magnitude_j]=size(magnitude);    %����magnitude����(ԭ����)�Ĵ�С

y1=zeros(magnitude_i,magnitude_j);             %%%%y1�����������ֵ������3db���������ص㱻�ָ������
y2=zeros(magnitude_i,magnitude_j);             %%%%y2�����������ֵ��3db������ֵ��20db���������ص㱻�ָ������

for i=1:width_i_3db
    y1(i_3db(i,1),j_3db(i,1))=1;                  %���3db��������ģ��
end
for i=1:width_i_20db
    y2(i_20db(i,1),j_20db(i,1))=1;                 %���20db��������ģ��
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
end                                              %%��ʱ�õ��ķ�ֵ����3db�����е㼯���ǰ��մӴ�С��˳�����е�

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
end                                              %%��ʱ�õ��ķ�ֵ����3db��20db�����е㼯���ǰ��մӴ�С��˳�����е�


plate=zeros(3,3);
R1=1;                              %R���ͬһ�ָ�����
for k=1:width_i_3db
    temp=0;
    if i_3db(k,1)~=1 & j_3db(k,1)~=1 & i_3db(k,1)~=magnitude_i & j_3db(k,1)~=magnitude_j
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
end                                                                    %%%%%�Է�ֵ����3db�����е���зָ�

             %%%%%%%%%%%%%%%%R1�ǳ�ʼʱ���ָ�����ֵ��3db��������������Ŀ������һ��������ĸ�����%%%%%%%%%%
                       %%%%%%%%%%%%%%%%%%%��Ϊ�����ڻ��������ʱ����һ�����ֵ�ĳЩ����ᱻ�ϲ������ĳһ��
                       %%%%%%%%%%%%%%%%%%%���µ������ɿռ������ں�����Ż��л�����                     %%%%%%%%%%%
            

plate=zeros(3,3);   
R2=1;
for k=1:width_i_20db
    temp=0;
    if i_20db(k,1)~=1 & j_20db(k,1)~=1 & i_20db(k,1)~=magnitude_i & j_20db(k,1)~=magnitude_j
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
 
y3=y1+y2;                                      %%%%%%y3�������д�����ֵ�㵽����ֵ��20db�����б�Ƿָ�����
for i=1:magnitude_i
    for j=1:magnitude_j
        if y1(i,j)~=0
            y1(i,j)=y1(i,j)-1;
        end
        if y2(i,j)~=0
            y2(i,j)=y2(i,j)-1;
        end
    end
end                                          %%%%%%%%%%%%%%%%���ѭ����Ŀ�Ľ�����Ϊ���ñ�������1��ʼ��ǣ���������  


for k=1:width_i_3db
    temp=0;
    if i_3db(k,1)~=1 & j_3db(k,1)~=1 & i_3db(k,1)~=magnitude_i & j_3db(k,1)~=magnitude_j
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
                     n=n+1;                                 %%%%%%n��������ʾ�ڸ������к�y1(i_3db(k,1),j_3db(k,1))��ֵ��ȵ����ص�ĸ���
                  end
              end
          end
          if n==0                                            %%%%%%n==0��ʾû�к�y1(i_3db(k,1),j_3db(k,1))��ֵ��ȵ����ص�
              max_plate=max(max(plate));
              if max_plate==0                             %%%%%%%%%%�������˵��y1(i_3db(k,1),j_3db(k,1))�Ǹ������ĵ㣬��������8���㶼Ϊ0��
                 y1(i_3db(k,1),j_3db(k,1))=0;
              else                                         %%%%%%%%�������˵��y1(i_3db(k,1),j_3db(k,1))����ˣ���ʱ���%%%%%%%%%
                                                          %%%%%%%y1(i_3db(k,1),j_3db(k,1))�����������ĳ��������                                               
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
end                                                     %%%%%%%%%%%%%%%%%%%%%%%��y1ģ������Ż�
     


for k=1:width_i_20db
    temp=0;
    if i_20db(k,1)~=1 & j_20db(k,1)~=1 & i_20db(k,1)~=magnitude_i & j_20db(k,1)~=magnitude_j
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
                     n=n+1;                                 %%%%%%n��������ʾ�ڸ������к�y2(i_20db(k,1),j_20db(k,1))��ֵ��ȵ����ص�ĸ���
                  end
              end
          end
          if n==0                                            %%%%%%n==0��ʾû�к�y2(i_20db(k,1),j_20db(k,1))��ֵ��ȵ����ص�
              max_plate=max(max(plate));
              if max_plate==0                             %%%%%%%%%%�������˵��y2(i_20db(k,1),j_20db(k,1))�Ǹ������ĵ㣬��������8���㶼Ϊ0��
%                  y2(i_20db(k,1),j_20db(k,1))=0;
              else                                         %%%%%%%%�������˵��y2(i_20db(k,1),j_20db(k,1))����ˣ���ʱ���%%%%%%%%%
                                                          %%%%%%%y2(i_20db(k,1),j_20db(k,1))�����������ĳ��������                                               
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
                                                       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%��y2ģ������Ż�

%%%%%%%%%%%%%%%%%%%%%%%%�������R1��R2�ڵ����Ĺ����п�����������ϲ�ʹ��ĳЩ��ֵ����ֿռ���������Ż�
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


%%%%%%%%%%�Ż����㷨���ϲ�ĳЩ����%%%%%%%%%
%%%%%%%%%%�ϲ���׼���ǣ���˳���ҵ�ͼ�����������������ǵ�����������������ڱ���������������������Ϊһ��������򣬷���������������������
%%%%%%%%%%����
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

    
  
    
    
               
               
               