# OMP-based ASC提取算法使用指南

## 概述

本项目实现了基于正交匹配追踪(OMP)算法的属性散射中心(ASC)提取算法，用于处理MSTAR SAR数据集。该实现包含了完整的数据预处理、ASC提取、性能评估和可视化功能。

## 文件结构

```
script/dataProcess/
├── step1_MSTAR2mat.m              # MSTAR原始数据转换为.mat格式
├── step2_MSTAR_mat2raw.m          # .mat文件转换为.raw格式（可选）
├── step3_OMP_ASC_extraction.m     # 主要的OMP-ASC提取算法
├── OMP_ASC_utils.m                # 辅助函数和高级功能
├── test_OMP_ASC.m                 # 测试脚本和性能评估
└── image_read.m                   # 已有的图像读取函数
└── create_R1_for_image_read.m     # 已有的数据转换函数
```

## 快速开始

### 1. 数据准备

首先确保MSTAR原始数据已放置在正确的目录中：

```matlab
% 运行数据预处理（如果尚未完成）
run('script/dataProcess/step1_MSTAR2mat.m');
```

### 2. 基本ASC提取

运行主要的ASC提取算法：

```matlab
% 运行OMP-based ASC提取
run('script/dataProcess/step3_OMP_ASC_extraction.m');
```

### 3. 测试和评估

使用测试脚本进行单文件测试：

```matlab
% 编辑 test_OMP_ASC.m 中的 test_mode
test_mode = 'single_file';
run('script/dataProcess/test_OMP_ASC.m');
```

## 详细使用方法

### 配置参数

主要算法参数可以在 `step3_OMP_ASC_extraction.m` 中调整：

```matlab
config.max_iterations = 40;        % OMP最大迭代次数
config.residual_threshold = 1e-6;  % 残差收敛阈值
config.image_size = [128, 128];    % 处理图像尺寸
config.position_samples = 80;      % 空间位置采样密度
config.freq_samples = 64;          % 频率采样数
config.angle_samples = 64;         % 角度采样数
```

### 测试模式

`test_OMP_ASC.m` 提供三种测试模式：

#### 1. 单文件测试 (`single_file`)
- 快速测试单个文件的ASC提取效果
- 生成详细的可视化结果
- 适合算法验证和调试

```matlab
test_mode = 'single_file';
```

#### 2. 参数扫描 (`parameter_sweep`)
- 自动测试不同参数组合
- 找到最优参数配置
- 生成参数影响分析图表

```matlab
test_mode = 'parameter_sweep';
```

#### 3. 性能对比 (`performance_test`)
- 对比不同算法配置的性能
- 评估处理时间、重建质量等指标
- 生成综合性能报告

```matlab
test_mode = 'performance_test';
```

## 算法特点

### 1. 物理字典构建
- 基于ASC物理模型构建过完备字典
- 包含位置、频率、角度等物理参数
- 支持七参数ASC模型（可选）

### 2. OMP核心算法
- 贪心算法选择最匹配的字典原子
- 正交化处理避免重复选择
- 自适应停止条件

### 3. 性能评估
- PSNR（峰值信噪比）
- SSIM（结构相似性）
- 压缩比和稀疏度
- 处理时间分析

## 输出结果

### 1. ASC参数文件
每个处理的文件将生成包含以下内容的 `.mat` 文件：
- `asc_params`: 提取的ASC参数
- `reconstructed_image`: 重建图像
- `residual_history`: 收敛历史
- `config`: 使用的算法配置

### 2. 可视化结果
- 原始图像 vs 重建图像对比
- ASC位置分布散点图
- 算法收敛曲线
- 性能指标摘要

### 3. 文本报告
- ASC参数详细列表
- 算法性能指标
- 处理时间统计

## 高级功能

### 1. 七参数ASC模型
使用更精确的物理模型：

```matlab
% 在 OMP_ASC_utils.m 中
[dictionary, param_grid] = build_seven_parameter_ASC_dictionary(config);
```

### 2. 改进的OMP算法
包含正则化和自适应停止：

```matlab
[selected_atoms, coefficients, residual_history, info] = improved_OMP(signal, dictionary, config);
```

### 3. 批量处理
处理整个数据集：

```matlab
summary = batch_process_and_summarize(input_dir, output_dir, config);
```

## 性能优化建议

### 1. 内存优化
- 对于大型字典，考虑分批处理
- 使用稀疏矩阵表示
- 适当减少采样密度

### 2. 计算优化
- 并行化字典构建过程
- 使用GPU加速（需要额外实现）
- 预计算常用字典

### 3. 参数调优
- 根据数据特点调整 `position_samples`
- 平衡 `max_iterations` 和处理时间
- 根据噪声水平设置 `residual_threshold`

## 常见问题解决

### 1. 内存不足
```matlab
% 减少字典大小
config.position_samples = 60;  % 从80减少到60
config.freq_samples = 32;      % 从64减少到32
```

### 2. 处理时间过长
```matlab
% 使用快速配置
config.max_iterations = 20;
config.image_size = [64, 64];  % 使用较小图像
```

### 3. 重建质量不佳
```matlab
% 增加迭代次数和采样密度
config.max_iterations = 50;
config.position_samples = 100;
config.residual_threshold = 1e-7;
```

## 与原有代码集成

本算法可以与现有的数据处理流程无缝集成：

1. 使用 `step1_MSTAR2mat.m` 预处理数据
2. 运行 `step3_OMP_ASC_extraction.m` 提取ASC
3. 结果可以输入到后续的目标识别或分类算法中

## 扩展和定制

### 1. 添加新的ASC模型
在 `OMP_ASC_utils.m` 中实现新的响应函数

### 2. 集成其他稀疏算法
替换OMP核心算法，如ISTA、AMP等

### 3. 添加深度学习后处理
结合深度展开网络进行参数优化

## 参考文献

1. Yang, H., Zhang, Z., & Huang, Z. "Interpretable attributed scattering center extracted via deep unfolding." IGARSS 2024.
2. Tropp, J. A., & Gilbert, A. C. "Signal recovery from random measurements via orthogonal matching pursuit." IEEE Transactions on Information Theory, 2007.
3. Gerry, M. J., Potter, L. C., Gupta, I. J., & Van Der Merwe, A. "A parametric model for synthetic aperture radar measurements." IEEE Transactions on Antennas and Propagation, 1999.

## 联系和支持

如有问题或建议，请：
1. 检查配置参数是否正确
2. 查看生成的日志和错误信息
3. 参考测试脚本中的示例用法
4. 调整算法参数以适应具体数据特点