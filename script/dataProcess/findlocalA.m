function A=findlocalA(image,x,y,a,r,o_o,L);






 fc=1e10;
 B=5e8;
 om=2.86;
 p=84;
 
 om=om*2*pi/360;         %%%%%%%%%%%%%%���ǶȻ�Ϊ����
 b=B/fc;
 fx1=(1-b/2)*fc;
 fx2=(1+b/2)*fc;
 fy1=-fc*sin(om/2);
 fy2=fc*sin(om/2);      %%%%%%%%%%%%%%ֱ������ϵ�������ȡֵ��Χ
 

 
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
    K=flipud(K);       %%%%%%%%%%%%%%%%%%%%�õ�ֱ������ϵ�µľ���
 
   T=taylorwin(p,3,-35);

       for j=1:p
           K(:,j)=K(:,j).*T;
       end



       for j=1:p
          K(j,:)=K(j,:).*T';
       end

 %%%%%%%%%%�Ӻ�����
 
 
   % 尺寸自适应：使 Z 与 image_interest 同尺寸
   global image_interest;
   global complex_temp;
   [q_rows, q_cols] = size(image_interest);
   p = min(p, min(q_rows, q_cols));
   Z=zeros(q_rows,q_cols);
%    Z(1+(q-p)/2:p+(q-p)/2,1+(q-p)/2:p+(q-p)/2)=K;
   Z(1:p,1:p)=K;
   
   Z=ifft2(Z);
   Z=ifftshift(Z);
   Z=Z.*image_interest;
   Z=abs(Z);
   image=reshape(image,q_rows*q_cols,1);
   Z=reshape(Z,q_rows*q_cols,1);
    A=Z\image;

%     A=(Z'*image)/(Z'*Z);


   