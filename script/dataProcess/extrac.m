% % 核心算法。
% 实现了论文中描述的“改进的 RD-AML-CLEAN”方法。
% 迭代地找到最亮的区域，调用其他函数进行分割、分类和参数估计，减去结果，并重复直到图像能量低于阈值 。

function scatter_all=extrac(image,image_complex);      %%%%%%%%%从image图像中提取散射中心，image_complex表示复图像，

close all;
 tic;

fc=1e10;
B=5e8;
om=2.86;
p=84;
q=128;

K=image;
K_complex=image_complex;
max_cell = max(max(image));
thresh=max_cell/10;                       %%%%%%%thresh是阈值（db值），表示当提取散射中心后所剩图像的最大值。
sim=zeros(q,q);

 
 om=om*2*pi/360;                      %%%%%%%%%%%        将角度转化成弧度，因为在MATLAB中，sin函数的自变量默认是弧度制的。
                                      %%%%%%%%%%%%%%%%%  另外，在经过test_signalandnoise过程后，om的值又变回了角度制，所以要再变一次
 
 b=B/fc;                                     
 fx1=(1-b/2)*fc;
 fx2=(1+b/2)*fc;
 fy1=-fc*sin(om/2);
 fy2=fc*sin(om/2); 
 
global complex_temp; 
global image_interest;                  %%%%%%%%%%%%%%这里定义两个全局变量，方便以后在迭代时可以调用


sca_s=1;
sca_all=1;
scatter_all=cell(p*p,1);
changeR=1;                              %%%%%%%%%%作为是否需要改变分割阈值的判定，如果为1则仍然按R1分割，如果为0则按R2分割


chazhi=max(max(K-sim));

K = TargetDetect( K,30 );


while chazhi>thresh


[y1 y2 R1 R2]=watershed_image(K);

if changeR==0
    y1=y2;
    R1=R2;
end

scatter=cell(p*p,1);
scatter_temp=scatter;
sca_s=1;


for sca_i=1:R1
    
    image_interest=ROI(y1,sca_i);             %%%%%%%%%%    image_interest是ROI区域模板
    image_temp  = image_interest.*K;     %%%%%%%%%%    image_temp是ROI区域模板对应的原图像像素值

    complex_temp= image_interest.*K_complex;   %%%%%%%%%%%     complex_temp是ROI区域模板对应的原图像复数信息





    [type,temp_coordinate]=selection(image_temp);     %%%%%%%%%%       type是该ROI区域的类型：若为0则是分布散射中心；若不为0则是局部散射中心
    % figure(2),imshow(image_temp);                     %%%%%%%%%%       temp_coordinate是该区域散射中心的位置，其单位是/像素
    
    

%%%%%%%%%%%%% 下面开始估计参数 %%%%%%%%%

if type~=-1

    complex_freq=fftshift(complex_temp);
    complex_freq=fft2(complex_freq);




    complex_freq=abs(complex_freq);

    complex_freq=complex_freq(1:p,1:p);




    
    opts=optimset('fmincon');
    %*****
    % options=optimset(opts,'DiffMaxChange',0.001,'LargeScale','off','HessUpdate','BFGS','LineSearchType','quadcubic','MaxFunEvals',1e20);
    %xulu修改上一句为：
    %修改版本2：
    options=optimset(opts,'DiffMaxChange',0.001,'LargeScale','off','HessUpdate','BFGS','MaxFunEvals',1e20);
    %****
    x=(temp_coordinate(1,2)-65)*0.3*84/128;
    y=(65-temp_coordinate(1,1))*0.3*84/128;         %%%%%%%%%%%该处容易搞混淆，因为对坐标的定义主观上和matlab中的不一样
end


    
        
    if(type~=0)
      if (type==1)

          r=0;
    
%%%%%%%%%%%%%   上面是单个局部散射中心需要迭代的3个参数  %%%%%%%%%%%
         global A;
        
         A=findlocalA(image_temp,x,y,0,0,0,0);

         [M1,fval(1),exitfig(1),output]=fmincon('extraction_local_a0',[x,y,0],[],[],[],[],[x-0.1,y-0.1,0],[x+0.1,y+0.1,1]);
         [M2,fval(2),exitfig(2),output]=fmincon('extraction_local_a05',[x,y,0],[],[],[],[],[x-0.1,y-0.1,0],[x+0.1,y+0.1,1]);
         [M3,fval(3),exitfig(3),output]=fmincon('extraction_local_a1',[x,y,0],[],[],[],[],[x-0.1,y-0.1,0],[x+0.1,y+0.1,1]);
         serial=find(fval==min(fval));
         a = ((serial)-1)/2;
     
         if serial==1
             M=M1;
         end
         if serial==2
             M=M2;
         end
         if serial==3
             M=M3;
         end


         if A==0
            if changeR==0
                K=K-image_temp;
            end
            if changeR==1
                changeR=0;
            end
         end
         
         

      if (type==-1)
          
      end

      scatter{sca_s,1}=[M(1) M(2) a M(3) 0 0 A];
      sca_s=sca_s+1;
      end
   end
    


    if(type==0)
        
        
          x_floor=floor(temp_coordinate(1,1));
          x_ceil=floor(temp_coordinate(1,1));
          y_floor =ceil(temp_coordinate(1,2));
          y_ceil =ceil(temp_coordinate(1,2));
          
          image_interest_copy=image_interest;
          image_interest_copy(:,y_floor)=1;
          image_interest_copy(:,y_ceil)=1;

          
          image_interest_copy=image_interest.*image_interest_copy;
          
          L_image_coordinate_floor = find(image_interest_copy(:,y_floor)==1);
          L_image_coordinate_ceil = find(image_interest_copy(:,y_ceil)==1);
          L_image_floor = max(L_image_coordinate_floor)-min(L_image_coordinate_floor);
          L_image_ceil = max(L_image_coordinate_ceil)-min(L_image_coordinate_ceil);
          L_image=(0.3*p/q)*max(L_image_floor,L_image_ceil);                               %%%L_image是分布散射中心在图像中长度   
          
          
          
            
        [fx_temp,fy_temp]=find(complex_freq==(max(max(complex_freq))));     %%%%%%  fx_temp代表第几行，fy_temp代表第几列
        fx_temp=fx_temp(1,1);
        fy_temp=fy_temp(1,1);
        fx_temp=fx_temp(1,1);
        fy_temp=fy_temp(1,1);
        fx_this=(fy_temp-1)*(B/(p-1))+fx1;
        fy_this=fy2-(fx_temp-1)*((2*fc*sin(om/2))/(p-1));
        o_o=atan(fy_this/fx_this);
        o_o=o_o*360/(2*pi);                    %%%%%%%%%%%%%  估计出o_o的初值
  
       %%%%%%%  下面是要估计L的初值  %%%%%%%%%%%%%
        slice_originate=complex_freq(:,fy_temp);               %%%%%%%%%%%%  slice_originate中存储的是原始的频域数据
        slice_one=slice_originate./(max(slice_originate));     %%%%%%%%%%%%  slice_one中存储的是归一化的原始数据             
        slice=slice_one;                                       %%%%%%%%%%%%  此时，slice和slice_one数组一样，存储的是归一化的频域数据
    
     %%%%%%%%%%   下面，是将slice中能够用来做二次拟合的数据提出来放到一起，将其它无用数据剔除
    
        cord = find (slice==1);
        for i=cord:-1:2
            if slice(i)<slice(i-1)
                slice(i-1)=0;
            end
        end
    
        for i=cord:(p-1)
            if slice(i)<slice(i+1)
                slice(i+1)=0;
            end
        end

        for i=1:p
            if slice(i)<=0.7
                slice(i)=0;
            end
        end                                                 
    
      %%%%%%%%%%%%%%%%%%%%%%   这3个循环的目的是为了找到slice数组中能够用来拟合的数据，将不符合条件的数据删去  
    
    slice=nonzeros(slice);
    size_slice=size(slice);
    slice=slice(1:(size_slice(1,1)));                        %%%%%%%%%%%        将第一个数据和最后一个数据去掉，称为“去头去尾”
 
    %%%%%%%%%%%%%%%   到此步骤为止，slice数组中数据已经变成了可以用来进行拟合的数据

  
    cord_1=find(slice==1);
    fx_cord_min=1+cord-cord_1;
    fx_cord_max=size_slice(1,1)+cord-cord_1;                 %%%%%%%%%%%%%%  在这里“去头去尾”是为了保证数据的可用性


    fx_cord=fx_cord_min:1:fx_cord_max;                         %%%%%%%%%%%%%%  fx_cord中存储的是slice_variable中数据在原数组中的位置

       
    
        fx_use=fx_this;
        fy_use=fy2-(fx_cord-1)*((2*fc*sin(om/2))/(p-1));         %%%%%%%%%%%%%%%%   fx_use和fy_use分别是slice中对应的fx和fy的频率值
        o_use=(atan(fy_use./fx_use))*360/(2*pi);      %%%%%%%%%%%%%%%%   o_use是角度值下的
        x_use=fx_use.*sin((o_use-o_o)*2*pi/360);       %%%%%%%%%%%%%%%%   由于o_o也是角度值下的，所以o_use和o_o的差值要转化成弧度值


        slice_fun=slice-1;
        x_use=x_use.*x_use;
        x_use=x_use';
        L_t=-x_use\slice_fun;
        L_t=sqrt(L_t*6);

        L=L_t*(3e8)/(2*pi);
   

   


        o_o_min=max(o_o-abs(0.5*o_o),(-om/2)*360/(2*pi));
        o_o_max=min(o_o+abs(0.5*o_o),(om/2)*360/(2*pi));
        L_min=L-0.3*L;
        L_max=L_image;
        x_min=x-0.1;
        x_max=x+0.1;
        y_min=y-0.1;
        y_max=y+0.1;
        [M4,fval_dis,exitfig(3),output]=fmincon('extraction_dis_a0',[x,y,o_o,L],[],[],[],[],[x_min,y_min,o_o_min,L_min],[x_max,y_max,o_o_max,L_max],'',options);

        
  
   
  if fval_dis>1
     hd_temp1= 0.58;
     hd_temp2= 0.57;
     hd_init1=3.899*hd_temp1*(1-hd_temp1);
     hd_init2=3.899*hd_temp2*(1-hd_temp2);
     hd1=hd_init1;
     hd2=hd_init2;
     o_o_init=o_o;
     L_init=L;
           for dis_i=1:100
               hd_inf=fval_dis;
               hd1=3.899*hd1*(1-hd1);
               hd2=3.899*hd2*(1-hd2);
               hd_o_o=(o_o_init-0.15)+0.3*hd1;
               hd_L  =(L_init-0.4)+0.8*hd2;
            if (hd_o_o>0)&(hd_L>0)
               m_hd  =[M4(1),M4(2),hd_o_o,hd_L];
               z_hd  =extraction_dis_a0(m_hd);
               if z_hd < hd_inf
                   o_o = hd_o_o;
                   L   = hd_L;
                   hd_inf=z_hd;
                        
                   o_o_min=max(o_o-abs(0.45*o_o),(-om/2)*360/(2*pi));
                   o_o_max=min(o_o+abs(0.45*o_o),(om/2)*360/(2*pi));
                   L_min=L-0.45*L;
                   L_max=L+0.45*L;
                   x_min=x-0.1;
                   x_max=x+0.1;
                   y_min=y-0.1;
                   y_max=y+0.1;
                   [M4,fval_dis,exitfig(3),output]=fmincon('extraction_dis_a0',[x,y,o_o,L],[],[],[],[],[x_min,y_min,o_o_min,L_min],[x_max,y_max,o_o_max,L_max],'',options);

                   if fval_dis<1
                      break;
                   end
               end
            end  
           end
  end
A=findlocalA(image_temp,M4(1),M4(2),0,0,M4(3),M4(4));
    

        
    a=finda(image_temp,A,x,y,0,o_o,L);

    scatter{sca_s,1}=[M4(1) M4(2) a 0 M4(3) M4(4) A];

         
     sca_s=sca_s+1;
    
    end

end



if sca_s~=1
  scatter_last=cell(sca_s-1,1);
  for i=1:sca_s-1
      scatter_last{i,1}=scatter{i,1};
  end

  scatter=scatter_last;

  sim=simulation(scatter_last);
  chazhi=max(max(K-sim));
  K=K-sim;


  for i=1:sca_s-1
      scatter_all{sca_all-1+i,1}=scatter{i,1};
  end

  sca_all=sca_s+sca_all-1;
  
  changeR=1;

end
    if sca_s==1
    
        if changeR==0
           image_temp = y1.*image;
           K=K-image_temp;
           chazhi=max(max(K));
           changeR=1;
        end
       
       if changeR==1      
          image_temp  = y1.*image; 
          chazhi=max(max(K-image_temp));
          changeR=0;
       end

    end
end

  scatter_all_copy=cell(sca_all-1,1);

  for i=1:sca_all-1
      scatter_all_copy{i,1}=scatter_all{i,1};
  end

  scatter_all=scatter_all_copy;


  toc;
























