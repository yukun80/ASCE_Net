function z=extraction_dis_a05(m); %%%%%%%%%fc������Ƶ�ʣ�B�Ǵ�����om�ǹ۲��
 %%%%%%%%%%%%%%���ø�����ģ�ͺ͹��ƵĲ�������,���õ�ģ����ֱ��Ƶ�����µ�ģ�͡������Ƶ��Ӵ�����֮��ľ���%%%%%%%%%%%%%

 
x=m(1);
y=m(2);
o_o=m(3)
L=m(4);
A=m(5);
r=0;
a=0.5;





    fc=1e10;
    B=5e8;
    om=2.86;
 
   om=om*2*pi/360;         %%%%%%%%%%%%%%���ǶȻ�Ϊ����
    b=B/fc;
    fx1=(1-b/2)*fc;
    fx2=(1+b/2)*fc;
    fy1=-fc*sin(om/2);
    fy2=fc*sin(om/2);      %%%%%%%%%%%%%%ֱ������ϵ�������ȡֵ��Χ
 
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
    K=flipud(K);       %%%%%%%%%%%%%%%%%%%%�õ�ֱ������ϵ�µľ���
 
   T=taylorwin(p,3,-35);

       for j=1:p
           K(:,j)=K(:,j).*T;
       end
       for j=1:p
           K(j,:)=K(j,:).*T';
       end

 %%%%%%%%%%�Ӻ�����
 

   Z=zeros(q,q);
%    Z(1+(q-p)/2:p+(q-p)/2,1+(q-p)/2:p+(q-p)/2)=K;
   Z(1:p,1:p)=K;

   global complex_temp;
   global image_interest;

   Z=ifft2(Z);
   Z=ifftshift(Z);
   Z=Z.*image_interest;

   Z1=Z-complex_temp;
   Z2=abs(Z1.*(Z1));
   z=sum(Z2(:));
%    Z1=reshape(Z1,q*q,1);
%    z=Z1*(Z1');


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

 