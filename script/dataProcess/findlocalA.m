function A=findlocalA(image,x,y,a,r,o_o,L);






 fc=1e10;
 B=5e8;
 om=2.86;
 p=84;
 q=128;
 
 om=om*2*pi/360;         %%%%%%%%%%%%%%将角度化为弧度
 b=B/fc;
 fx1=(1-b/2)*fc;
 fx2=(1+b/2)*fc;
 fy1=-fc*sin(om/2);
 fy2=fc*sin(om/2);      %%%%%%%%%%%%%%直角坐标系下两轴的取值范围
 

 
    K=zeros(1,p*p);
    i=1;
    s=0;
    for fx=fx1:B/(p-1):fx2
        for fy=fy1:((2*fc*sin(om/2))/(p-1)):fy2
            K(1,i)=model_rightangle(om,fx,fy,fc,x,y,a,r,o_o,L,1);
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

   global image_interest;
   global complex_temp;
   
   Z=ifft2(Z);
   Z=ifftshift(Z);
   Z=Z.*image_interest;
   Z=abs(Z);
   image=reshape(image,q*q,1);
   Z=reshape(Z,q*q,1);
    A=Z\image;

%     A=(Z'*image)/(Z'*Z);


   