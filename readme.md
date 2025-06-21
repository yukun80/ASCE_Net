# 项目文件结构

```
ASCE_Net/
├── .git/
├── datasets/
│   └── SAR_ASC_Project/
│       ├── 03_Training_ASC_reconstruct_jpeg/
│       ├── 03_Training_ASC_reconstruct/
│       ├── 03_Training_ASC/
│       ├── 02_Data_Processed_jpg_tmp/
│       ├── 02_Data_Processed_raw/
│       ├── 01_Data_Processed_mat/
│       └── 00_Data_Raw/
├── script/
│   ├── dataProcess/
│   │   ├── step3_main_xulu.m
│   │   ├── step2_MSTAR_mat2raw.m
│   │   ├── step1_MSTAR2mat.m
│   │   ├── TargetDetect.m
│   │   ├── visualize_results.m
│   │   ├── step35_Output_Difference.m
│   │   ├── model_rightangle.m
│   │   ├── selection.m
│   │   ├── watershed_image.m
│   │   ├── extrac.m
│   │   ├── image_read.m
│   │   ├── create_R1_for_image_read.m
│   │   ├── MSTAR2JPG.m
│   │   ├── notes+source.txt
│   │   ├── mstar_to_raw_for_image_read.m
│   │   ├── findlocalA.m
│   │   ├── finda.m
│   │   ├── extraction_local_xy_a05.m
│   │   ├── extraction_local_xy_a1.m
│   │   ├── extraction_local_xy_a0.m
│   │   ├── extraction_local_xy.m
│   │   ├── extraction_local_a05.m
│   │   ├── extraction_local_a1.m
│   │   ├── extraction_local_a0.m
│   │   ├── extraction_dis_a05.m
│   │   ├── extraction_dis_a0.m
│   │   ├── spotlight.m
│   │   ├── simulation.m
│   │   ├── normxcorr2.m
│   │   ├── ROI.m
│   │   ├── label.m
│   │   ├── taylorwin.m
│   ├── reconstruct_img.py
│   └── tmp.py
├── .gitignore

the dataset structure is show blow:
datasets/SAR_ASC_Project/
├── 03_Training_ASC/           # MAT文件
│   ├── train_17_deg/
│   │   ├── T72/
│   │   │   ├── SN_132/        # <- SN子目录
│   │   │   │   ├── HB04001.015_yang.mat
│   │   │   │   └── HB04002.015_yang.mat
│   │   │   └── SN_812/
│   │   └── BTR70/...
│   └── test_15_deg/...
├── 04_MSTAR_ASC_LABELS/       # NPY文件（处理后）
│   ├── train_17_deg/
│   │   ├── T72/
│   │   │   ├── SN_132/        # <- 相同的SN子目录结构
│   │   │   │   ├── HB04001.015_5ch.npy
│   │   │   │   └── HB04002.015_5ch.npy
│   │   │   └── SN_812/
└── 02_Data_Processed_jpg/     # JPG文件（如果有）
    └── ... (相同结构)
```
