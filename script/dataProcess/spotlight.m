function [Z,s]=spotlight(fc,B,om,x,y,a,r,o_o,L,A, p_in, q_rows, q_cols) %%%%%%%%%fc������Ƶ�ʣ�B�Ǵ�����om�ǹ۲��
 %%%%%%%%%%%%%%���ø�����ģ�ͺ͹��ƵĲ�������,���õ�ģ����ֱ��Ƶ�����µ�ģ�͡������Ƶ��Ӵ�����֮��ľ���%%%%%%%%%%%%%

 om=om*2*pi/360;         %%%%%%%%%%%%%%���ǶȻ�Ϊ����
 b=B/fc;
 fx1=(1-b/2)*fc;
 fx2=(1+b/2)*fc;
 fy1=-fc*sin(om/2);
 fy2=fc*sin(om/2);      %%%%%%%%%%%%%%ֱ������ϵ�������ȡֵ��Χ
 
 % 使用外部传入的频域网格与输出尺寸
 if nargin < 11 || isempty(p_in)
     % 若未传入 p，则使用 84 基线，后续由调用侧按比例传入
     p_in = 84;
 end
 if nargin < 12 || isempty(q_rows)
     q_rows = 128;
 end
 if nargin < 13 || isempty(q_cols)
     q_cols = q_rows;
 end
 p = p_in;


 
  
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
 s=s/(p*p);      %%%%%%s������
       
 K=reshape(K,p,p);
 K=flipud(K);       %%%%%%%%%%%%%%%%%%%%�õ�ֱ������ϵ�µľ���
 
T=taylorwin(p,3,-35);   



    for j=1:p
        K(:,j)=K(:,j).*T;
    end



    for j=1:p
        K(j,:)=K(j,:).*T';
    end

 %%%%%%%%%%��̩�մ�
 
  
 Z=zeros(q_rows,q_cols);
 Z(1:p,1:p)=K;

 
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

 