# SAR图像对比度增强功能说明

## 功能概述

基于您提供的MATLAB脚本分析（特别是`step2_MSTAR_mat2raw.m`），我已经在`script/visualize_comprehensive.py`中实现了SAR图像对比度增强功能，模拟MATLAB的`imadjust`效果。

## 实现原理

### MATLAB原始方法对比
MATLAB脚本中有两种JPG生成方式：
```matlab
% V1版本：带对比度调整
imwrite(uint8(imadjust(fileimage) * 255), output_jpg_v1);

% V2版本：简单归一化  
imwrite(uint8(fileimage / max(fileimage(:)) * 255), output_jpg_v2);
```

### Python实现
我实现了对应的Python版本：

**V1方法（推荐）：imadjust风格增强**
1. **对比度拉伸**：使用2%和98%分位数进行动态范围调整
2. **自适应直方图均衡化**：增强局部对比度，clip_limit=0.03

**V2方法：简单归一化**
- 直接归一化到[0,1]范围

## 使用方法

### 1. 安装依赖
```bash
pip install scikit-image
```

### 2. 测试对比度增强效果
```bash
cd script
python test_contrast_enhancement.py
```

### 3. 运行完整的可视化程序
```bash
cd script
python visualize_comprehensive.py
```

## 增强效果

- ✅ **更高的对比度**：细节更清晰可见
- ✅ **更好的亮度**：图像整体更亮，类似V1.JPG效果
- ✅ **保持细节**：不会丢失重要的散射中心信息
- ✅ **统一处理**：所有SAR图像（原始、重建、误差图）都使用相同增强

## 核心函数

### `enhance_sar_contrast(image_array, method='v1')`
对单个图像数组进行对比度增强

### `prepare_sar_image_for_display(image_data, enhancement_method='v1')`  
准备SAR图像用于显示，包括复数处理和对比度增强

## 可视化改进

在`visualize_comprehensive.py`中，以下部分都应用了对比度增强：

1. **第一行**：原始SAR图像、真值重建、预测重建
2. **第三行**：检测结果叠加图、误差分析图
3. **检测函数**：散射中心检测可视化

## 对比效果

运行测试程序后，您将看到三种处理方式的对比：
- **Original**: 简单归一化（较暗）
- **Enhanced V1**: imadjust风格增强（推荐，更亮更清晰）
- **Enhanced V2**: 归一化方法

V1方法产生的效果应该与您提供的V1.JPG示例相似，具有更好的视觉效果。

## 技术细节

- 使用`skimage.exposure`模块实现专业的图像增强
- `rescale_intensity`进行对比度拉伸
- `equalize_adapthist`进行自适应直方图均衡化
- 参数经过调优，适合SAR图像特性

这个实现确保了SAR图像在Python中的可视化效果与MATLAB处理的V1.JPG版本相当，提供了更好的观察体验。 