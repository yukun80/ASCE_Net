function a=finda(image,A,x,y,r,o_o,L);





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
 

for a=0:0.5:1  
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
 

  % 尺寸自适应：使 Z 与 image_interest 同尺寸
  global complex_temp;
  global image_interest;
  [q_rows, q_cols] = size(image_interest);
  p = min(p, min(q_rows, q_cols));
  Z=zeros(q_rows,q_cols);
  Z(1:p,1:p)=K;

   Z=ifft2(Z);
   Z=ifftshift(Z);
   Z=Z.*image_interest;
   Z=real(Z);
  complex_temp_copy=real(complex_temp);
   
  Z1=Z-complex_temp_copy;
   Z2=abs(Z1.*(Z1));
 
%    z((a+1)/0.5+1)=sum(Z2(:));
   z(a*2+1)=sum(Z2(:));        %%%%%%%%  aֻȡ3��ֵ
   
end

% a=find(z==min(z))*0.5-1.5;      %%%%%%%%  aȡ5��ֵ
  a=(find(z==min(z))-1)/2;      %%%%%%%%  aֻȡ3��ֵ
   
   