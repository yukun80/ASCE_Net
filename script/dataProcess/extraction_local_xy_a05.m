function z=extraction_local_xy_a05(m); %%%%%%%%%fc是中心频率，B是带宽，om是观测角
 %%%%%%%%%%%%%%利用给定的模型和估计的参数成像,所用的模型是直角频率域下的模型。输出是频域加窗补零之后的矩阵%%%%%%%%%%%%%

 
x=m(1);
y=m(2);
% A=m(3);
r=0;
o_o=0.5;
L=0;
a=0.5;

global A;



    fc=1e10;
    B=5e8;
    om=2.86;
 
   om=om*2*pi/360;         %%%%%%%%%%%%%%将角度化为弧度
    b=B/fc;
    fx1=(1-b/2)*fc;
    fx2=(1+b/2)*fc;
    fy1=-fc*sin(om/2);
    fy2=fc*sin(om/2);      %%%%%%%%%%%%%%直角坐标系下两轴的取值范围
 
    p=84;
    q=128;


 
  
    K=zeros(1,p*p);
    i=1;
    for fx=fx1:B/(p-1):fx2
        for fy=fy1:((2*fc*sin(om/2))/(p-1)):fy2
            K(1,i)=model_rightangle(om,fx,fy,fc,x,y,a,r,o_o,L,A);
            i=i+1;
        end
    end
       
    K=reshape(K,p,p);
    K=flipud(K);       %%%%%%%%%%%%%%%%%%%%得到直角坐标系下的矩阵
 
   T=taylorwin(p,3,-35);

       for j=1:p
           K(:,j)=K(:,j).*T;
       end
       for j=1:p
           K(j,:)=K(j,:).*T';
       end

 %%%%%%%%%%加汉宁窗
 

   Z=zeros(q,q);
%    Z(1+(q-p)/2:p+(q-p)/2,1+(q-p)/2:p+(q-p)/2)=K;
   Z(1:p,1:p)=K;

   global complex_temp;
   global image_interest;

   Z=ifft2(Z);
   Z=ifftshift(Z);
   Z=Z.*image_interest;

   Z1=Z-complex_temp;
   
   Z1=reshape(Z1,1,q*q);
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
% xlabel('距离向');
% ylabel('方位向');





%%%%%%%%%%%%%%补零并成像


%m=fx1:B/(p-1):fx2;
%n=fy1:((2*fc*sin(om/2))/(p-1)):fy2;
%figure(2),mesh(m,n,Z);
%xlabel('fx');
%ylabel('fy');
%zlabel('E(fx,fy)');

 