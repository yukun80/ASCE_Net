% % ????????
% ??????????????????????? RD-AML-CLEAN????????
% ?????????????????????????????????з????????????????????????????????????????????? ??

function scatter_all=extrac(image,image_complex);      %%%%%%%%%??image????????????????image_complex????????

close all;
 tic;

fc=1e10;
B=5e8;
om=2.86;
p=84;
q=128;

K=image;
K_complex=image_complex;
% 动态适配图像与频域尺寸
[q_rows, q_cols] = size(K);
q = q_rows; % 频域参考（行数），若为非正方图也可使用列数或二者平均
qref = min(q_rows, q_cols);
% 确保频率网格尺寸不超过图像最小边
p = min(p, qref);
max_cell = max(max(image));
thresh=max_cell/10;                       %%%%%%%thresh???????db??????????????????????????????????
sim=zeros(size(K));

 
 om=om*2*pi/360;                      %%%%%%%%%%%        ???????????????????MATLAB?У?sin???????????????????????
                                      %%%%%%%%%%%%%%%%%  ?????????test_signalandnoise?????om?????????????????????????
 
 b=B/fc;                                     
 fx1=(1-b/2)*fc;
 fx2=(1+b/2)*fc;
 fy1=-fc*sin(om/2);
 fy2=fc*sin(om/2); 
 
global complex_temp; 
global image_interest;                  %%%%%%%%%%%%%%??????????????????????????????????????


sca_s=1;
sca_all=1;
scatter_all=cell(p*p,1);
changeR=1;                              %%%%%%%%%%????????????????????ж???????1???????R1???????0??R2???


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
    
    image_interest=ROI(y1,sca_i);             %%%%%%%%%%    image_interest??ROI???????
    image_temp  = image_interest.*K;     %%%%%%%%%%    image_temp??ROI????????????????????

    complex_temp= image_interest.*K_complex;   %%%%%%%%%%%     complex_temp??ROI????????????????????





    [type,temp_coordinate]=selection(image_temp);     %%%%%%%%%%       type???ROI?????????????0???????????????????0?????????????
    % figure(2),imshow(image_temp);                     %%%%%%%%%%       temp_coordinate???????????????λ?????λ??/????
    
    

%%%%%%%%%%%%% ???濪???????? %%%%%%%%%

if type~=-1

    complex_freq=fftshift(complex_temp);
    complex_freq=fft2(complex_freq);




    complex_freq=abs(complex_freq);

    complex_freq=complex_freq(1:p,1:p);




    
    opts=optimset('fmincon');
    %*****
    % options=optimset(opts,'DiffMaxChange',0.001,'LargeScale','off','HessUpdate','BFGS','LineSearchType','quadcubic','MaxFunEvals',1e20);
    %xulu???????????
    %???汾2??
    options=optimset(opts,'DiffMaxChange',0.001,'LargeScale','off','HessUpdate','BFGS','MaxFunEvals',1e20);
    %****
    % 自适应坐标换算（以图像中心为参考，随尺寸缩放）
    center_row = floor(q_rows/2) + 1;
    center_col = floor(q_cols/2) + 1;
    scale = 0.3 * p / qref;
    x=(temp_coordinate(1,2)-center_col) * scale;
    y=(center_row-temp_coordinate(1,1)) * scale;         %%%%%%%%%%%?????????????????????????????????matlab?е?????
end


    
        
    if(type~=0)
      if (type==1)

          r=0;
    
%%%%%%%%%%%%%   ????????????????????????????3??????  %%%%%%%%%%%
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
         L_image=(0.3*p/qref)*max(L_image_floor,L_image_ceil);                               %%%L_image?????????????????г???   
          
          
          
            
        [fx_temp,fy_temp]=find(complex_freq==(max(max(complex_freq))));     %%%%%%  fx_temp????????У?fy_temp?????????
        fx_temp=fx_temp(1,1);
        fy_temp=fy_temp(1,1);
        fx_temp=fx_temp(1,1);
        fy_temp=fy_temp(1,1);
        fx_this=(fy_temp-1)*(B/(p-1))+fx1;
        fy_this=fy2-(fx_temp-1)*((2*fc*sin(om/2))/(p-1));
        o_o=atan(fy_this/fx_this);
        o_o=o_o*360/(2*pi);                    %%%%%%%%%%%%%  ?????o_o????
  
       %%%%%%%  ???????????L????  %%%%%%%%%%%%%
        slice_originate=complex_freq(:,fy_temp);               %%%%%%%%%%%%  slice_originate?д洢???????????????
        slice_one=slice_originate./(max(slice_originate));     %%%%%%%%%%%%  slice_one?д洢???ǹ????????????             
        slice=slice_one;                                       %%%%%%%%%%%%  ?????slice??slice_one??????????洢???ǹ?????????????
    
     %%%%%%%%%%   ???棬???slice??????????????????????????????????????????????????
    
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
    
      %%%%%%%%%%%%%%%%%%%%%%   ??3??????????????????slice??????????????????????????????????????????  
    
    slice=nonzeros(slice);
    size_slice=size(slice);
    slice=slice(1:(size_slice(1,1)));                        %%%%%%%%%%%        ???????????????????????????????????β??
 
    %%%%%%%%%%%%%%%   ???????????slice?????????????????????????????????????

  
    cord_1=find(slice==1);
    fx_cord_min=1+cord-cord_1;
    fx_cord_max=size_slice(1,1)+cord-cord_1;                 %%%%%%%%%%%%%%  ?????????β???????????????????


    fx_cord=fx_cord_min:1:fx_cord_max;                         %%%%%%%%%%%%%%  fx_cord?д洢????slice_variable??????????????е?λ??

       
    
        fx_use=fx_this;
        fy_use=fy2-(fx_cord-1)*((2*fc*sin(om/2))/(p-1));         %%%%%%%%%%%%%%%%   fx_use??fy_use?????slice?ж????fx??fy??????
        o_use=(atan(fy_use./fx_use))*360/(2*pi);      %%%%%%%%%%%%%%%%   o_use????????
        x_use=fx_use.*sin((o_use-o_o)*2*pi/360);       %%%%%%%%%%%%%%%%   ????o_o??????????????o_use??o_o??????????????


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

  sim=simulation(scatter_last, q_rows, q_cols);
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
























