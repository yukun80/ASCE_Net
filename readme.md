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
SAR_ASC_Project/
├── 00_Data_Raw/                          # 原始MSTAR数据
├── 01_Data_Processed_mat/                # 预处理后的MAT文件
├── 02_Data_Processed_raw/                # 预处理后的RAW文件
│   ├── train_17_deg/                     # 训练数据(17度)
│   │   ├── BMP2/                         # BMP2目标类别
│   │   ├── BTR70/                        # BTR70目标类别  
│   │   └── T72/                          # T72目标类别
│   │       ├── SN_132/                   # 序列号子目录
│   │       ├── SN_812/                   # 序列号子目录
│   │       └── SN_S7/                    # 序列号子目录
│   └── test_15_deg/                      # 测试数据(15度)
│       ├── BMP2/                         # BMP2目标类别
│       ├── BTR70/                        # BTR70目标类别
│       └── T72/                          # T72目标类别
├── 02_Data_Processed_jpg/                # JPG格式处理数据
├── 03_Training_ASC/                      # ASC训练数据(MAT格式)
│   ├── train_17_deg/                     # 训练数据(17度)
│   │   ├── BMP2/                         # BMP2目标类别
│   │   ├── BTR70/                        # BTR70目标类别
│   │   └── T72/                          # T72目标类别
│   │       ├── SN_132/                   # 序列号子目录
│   │       ├── SN_812/                   # 序列号子目录
│   │       └── SN_S7/                    # 序列号子目录
│   └── test_15_deg/                      # 测试数据(15度)
│       ├── BMP2/                         # BMP2目标类别
│       ├── BTR70/                        # BTR70目标类别
│       └── T72/                          # T72目标类别
├── 03_Training_ASC_reconstruct/          # ASC重构数据
├── 03_Training_ASC_reconstruct_jpeg/     # ASC重构JPEG数据
├── 04_MSTAR_ASC_LABELS/                  # 深度学习标签数据(NPY格式)
│   ├── train_17_deg/                     # 训练标签(17度)
│   │   ├── BMP2/                         # BMP2目标类别标签
│   │   ├── BTR70/                        # BTR70目标类别标签
│   │   └── T72/                          # T72目标类别标签
│   │       ├── SN_132/                   # 序列号子目录标签
│   │       ├── SN_812/                   # 序列号子目录标签
│   │       └── SN_S7/                    # 序列号子目录标签
│   └── test_15_deg/                      # 测试标签(15度)
│       ├── BMP2/                         # BMP2目标类别标签
│       ├── BTR70/                        # BTR70目标类别标签
│       └── T72/                          # T72目标类别标签
├── tmp_Data_Processed_raw/               # 临时RAW数据(测试用)
├── tmp_MSTAR_ASC_LABELS/                 # 临时标签数据(测试用)
├── tmp_Training_ASC/                     # 临时ASC数据(测试用)
└── validation_outputs/                   # 验证输出结果
```
