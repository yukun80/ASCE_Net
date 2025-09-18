function z=extraction_local_xy_a1(m); %%%%%%%%%fc������Ƶ�ʣ�B�Ǵ�����om�ǹ۲��
 %%%%%%%%%%%%%%���ø�����ģ�ͺ͹��ƵĲ�������,���õ�ģ����ֱ��Ƶ�����µ�ģ�͡������Ƶ��Ӵ�����֮��ľ���%%%%%%%%%%%%%

 
x=m(1);
y=m(2);
% A=m(3);
r=0.5;
o_o=0;
L=0;
a=1;

global A;



    fc=1e10;
    B=5e8;
    om=2.86;
 
   om=om*2*pi/360;         %%%%%%%%%%%%%%���ǶȻ�Ϊ����
    b=B/fc;
    fx1=(1-b/2)*fc;
    fx2=(1+b/2)*fc;
    fy1=-fc*sin(om/2);
    fy2=fc*sin(om/2);      %%%%%%%%%%%%%%ֱ������ϵ�������ȡֵ��Χ
 
    % 在构建 K 之前确定 p（按当前 ROI 尺寸比例缩放）
    global complex_temp;
    [q_rows, q_cols] = size(complex_temp);
    p = max(4, round(84 * min(q_rows, q_cols) / 128));
    q = q_rows;


 
  
    K=zeros(1,p*p);
    i=1;
    for fx=fx1:B/(p-1):fx2
        for fy=fy1:((2*fc*sin(om/2))/(p-1)):fy2
            K(1,i)=model_rightangle(om,fx,fy,fc,x,y,a,r,o_o,L,A);
            i=i+1;
        end
    end
       
    K=reshape(K,p,p);
    K=flipud(K);       %%%%%%%%%%%%%%%%%%%%�õ�ֱ������ϵ�µľ���
 
   T=taylorwin(p,3,-35);

       for j=1:p
           K(:,j)=K(:,j).*T;
       end
       for j=1:p
           K(j,:)=K(j,:).*T';
       end

 %%%%%%%%%%�Ӻ�����
 

   Z=zeros(q_rows,q_cols);
%    Z(1+(q-p)/2:p+(q-p)/2,1+(q-p)/2:p+(q-p)/2)=K;
   Z(1:p,1:p)=K;

   global complex_temp;
   global image_interest;

   Z=ifft2(Z);
   Z=ifftshift(Z);
   Z=Z.*image_interest;

  Z1=Z-complex_temp;
  
  Z1=reshape(Z1,1,q_rows*q_cols);
   Z2=abs(Z1*(Z1)');
%    Z2=Z1*(Z1)';
%    z=norm(Z2,2);
   z=sum(Z2(:));



% z=sum(Z2(:));
% Z2=abs(Z1.^2);
% z=sum(Z2(:));
% figure(3),imshow(abs(Z));

 
% Z=ifft2(Z);
% Z=ifftshift(Z);
% Z=abs(Z);
% imshow(Z);
% xlabel('������');
% ylabel('��λ��');





%%%%%%%%%%%%%%���㲢����


%m=fx1:B/(p-1):fx2;
%n=fy1:((2*fc*sin(om/2))/(p-1)):fy2;
%figure(2),mesh(m,n,Z);
%xlabel('fx');
%ylabel('fy');
%zlabel('E(fx,fy)');

 