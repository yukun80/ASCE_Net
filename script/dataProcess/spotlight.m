function [Z,s]=spotlight(fc,B,om,x,y,a,r,o_o,L,A); %%%%%%%%%fc是中心频率，B是带宽，om是观测角
 %%%%%%%%%%%%%%利用给定的模型和估计的参数成像,所用的模型是直角频率域下的模型。输出是频域加窗补零之后的矩阵%%%%%%%%%%%%%

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
 s=0;
 for fx=fx1:B/(p-1):fx2
     for fy=fy1:((2*fc*sin(om/2))/(p-1)):fy2
         K(1,i)=model_rightangle(om,fx,fy,fc,x,y,a,r,o_o,L,A);
         s=abs((K(1,i))^2)+s;
         i=i+1;
     end
 end
 s=s/(p*p);      %%%%%%s是能量
       
 K=reshape(K,p,p);
 K=flipud(K);       %%%%%%%%%%%%%%%%%%%%得到直角坐标系下的矩阵
 
T=taylorwin(p,3,-35);   



    for j=1:p
        K(:,j)=K(:,j).*T;
    end



    for j=1:p
        K(j,:)=K(j,:).*T';
    end

 %%%%%%%%%%加泰勒窗
 

Z=zeros(q,q);
Z(1:p,1:p)=K;

 
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

 