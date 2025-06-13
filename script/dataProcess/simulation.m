function  K=simulation(T);

fc=1e10;
B=5e8;
om=2.86;

 
 p=84;
 q=128;


scat=size(T);
scat=scat(1,1);




 
  
 K=zeros(q,q);
 K_temp=zeros(q,q);
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
     [K_temp,s_temp]=spotlight(fc,B,om,x,y,a,r,o_o,L,A);
     K=K+K_temp;
     K_temp=zeros(q,q);
     s=s+s_temp;
 end

%%%%%加高斯白噪声%%%%%%%%


% K_freq=wgn(q,q,s_freq);
% K=awgn(K,3,'measured');

 

K=ifft2(K);
K=ifftshift(K);
K_complex=K;      %%%%%%%% K_complex保存的是图象复数据 %%%%%%%%%
K=abs(K);
% imshow(K);
xlabel('距离向');
ylabel('方位向');
