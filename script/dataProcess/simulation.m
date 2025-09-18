function  K=simulation(T, q_rows, q_cols)

fc=1e10;
B=5e8;
om=2.86;

 
 p=84;
 % 动态输出尺寸，默认正方形时使用 q_rows
 if nargin < 2 || isempty(q_rows)
     q_rows = 128;
 end
 if nargin < 3 || isempty(q_cols)
     q_cols = q_rows;
 end
 % 频域网格维持与 128 的比例：p = round(84 * min(q_rows,q_cols) / 128)
 p = round(84 * min(q_rows, q_cols) / 128);


scat=size(T);
scat=scat(1,1);




 
  
 K=zeros(q_rows,q_cols);
 K_temp=zeros(q_rows,q_cols);
 i=1;
 s=0;
 for j=1:scat
     W=T{j,1};
     x=W(1,1);
     y=W(1,2);
     a=W(1,3);
     r=W(1,4);
     o_o=W(1,5);
     L=W(1,6);
     A=W(1,7);
     [K_temp,s_temp]=spotlight(fc,B,om,x,y,a,r,o_o,L,A, p, q_rows, q_cols);
     K=K+K_temp;
     K_temp=zeros(q_rows,q_cols);
     s=s+s_temp;
 end

%%%%%�Ӹ�˹������%%%%%%%%


% K_freq=wgn(q,q,s_freq);
% K=awgn(K,3,'measured');

 

K=ifft2(K);
K=ifftshift(K);
K_complex=K;      %%%%%%%% K_complex�������ͼ������ %%%%%%%%%
K=abs(K);
% imshow(K);
xlabel('������');
ylabel('��λ��');
