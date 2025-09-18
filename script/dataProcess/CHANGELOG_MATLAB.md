## MATLAB 代码修改日志（尺寸自适配与稳健性改造）

日期: 2025-09-18

### 背景与目标
- 适配非 128×128 图像尺寸（如 158×158），消除尺寸硬编码导致的报错。
- 通过“按 128 基线同比例缩放”的策略，保持算法在不同分辨率下的相对行为一致。
- 修复索引与尺寸不匹配问题，提高流程稳健性。

### 核心策略与约定
- 频域采样网格尺寸 p：按 qref = min(H, W) 比例缩放
  - p = round(84 × qref / 128)
  - 例：158×158 → p ≈ round(84×158/128) = 104
- 目标检测窗口 winsize：winsize = round(30 × qref / 128)
- 形态学圆盘半径：se_radius = round(4 × qref / 128)
- 坐标换算：以图像中心为基准、缩放因子 scale = 0.3 × p / qref
- 所有数组切片索引均保证为整数，并做边界保护

---

### 主要变更（按模块）

#### 一、数据转换与预览
- `script/dataProcess/step2_MSTAR_mat2raw.m`
  - 在调用转换前 `load(...,'Img')` 读取 `[rows, cols]`，动态构造 RAW 文件名：`sprintf('%s.%dx%d.raw', ..., cols, rows)`。
  - 预览阶段不再硬编码 `.128x128.raw`，改为使用上述动态文件名。
- `script/dataProcess/create_R1_for_image_read.m`
  - 放宽输入尺寸校验：仅要求 `Img` 与 `phase` 尺寸一致，不再强制 128×128。
  - 输出文件名包含尺寸后缀 `.%dx%d.raw`，数据按 float32、大端（big-endian）写。
- `script/dataProcess/image_read.m`
  - 兼容任意尺寸（从文件名解析）。建议按需改为大端读取（`'b'`）以匹配写入端（当前保留原状，仅记录此建议）。

#### 二、主流程与尺寸传递
- `script/dataProcess/step3_main_xulu.m`
  - 读取 `fileimage` 后获取 `[h, w]`，调用 `simulation(scatter_all, h, w)`，确保重建图尺寸与原图一致。
  - 结果 `diff = fileimage - s` 不再出现尺寸不匹配。
- `script/dataProcess/extrac.m`
  - `sim = zeros(size(K))`，消除 `K - sim` 尺寸不匹配。
  - 基于 `qref = min(size(K))` 计算 `p = round(84 × qref / 128)`。
  - 检测窗口自适应：`winsize = round(30 × qref / 128)` 传入 `TargetDetect`。
  - 坐标换算改为以图像中心为参考，缩放随 p/qref 而变。
  - 调用重建：`simulation(scatter_last, q_rows, q_cols)`。

#### 三、重建与频域核
- `script/dataProcess/simulation.m`
  - 函数签名变更：`simulation(T, q_rows, q_cols)`。
  - 输出尺寸由入参指定；频域网格 `p = round(84 × min(q_rows,q_cols) / 128)`。
  - 调用 `spotlight(..., p, q_rows, q_cols)`。
- `script/dataProcess/spotlight.m`
  - 函数签名变更：`spotlight(fc,B,om,x,y,a,r,o_o,L,A, p_in, q_rows, q_cols)`。
  - 生成 `Z` 尺寸为 `[q_rows, q_cols]`，将 `K(1:p,1:p)` 放置在左上角。

#### 四、检测与分割
- `script/dataProcess/TargetDetect.m`
  - `localsize = floor(winsize/2)`，确保整数索引；所有切片加 `max(1, ...)`、`min(...)` 的边界保护。
  - 形态学圆盘半径自适应：`se_radius = round(4 × min(row,col) / 128)`。
- `script/dataProcess/watershed_image.m`
  - 将涉及边界的硬编码 `128` 改为动态 `magnitude_i/magnitude_j`。

#### 五、参数估计（局部/分布/位置等）
- 以下函数统一了逻辑：在“构建频域核 K 之前”先按比例计算 p，并将 `Z` 构造为与 `image_interest` 或 `complex_temp` 同尺寸；`reshape` 使用 `q_rows*q_cols`。
  - `script/dataProcess/findlocalA.m`
  - `script/dataProcess/finda.m`
  - `script/dataProcess/extraction_local_a0.m`
  - `script/dataProcess/extraction_local_a05.m`
  - `script/dataProcess/extraction_local_a1.m`
  - `script/dataProcess/extraction_local_xy.m`
  - `script/dataProcess/extraction_local_xy_a0.m`
  - `script/dataProcess/extraction_local_xy_a1.m`
  - `script/dataProcess/extraction_dis_a0.m`

> 修复的典型问题：
> - “无法执行赋值，因为左侧大小为 p×p，右侧大小为 84×84” → 现已在创建 K 前确定 p。
> - “冒号运算符需要整数操作数” → 统一使用 `floor/round` 并做好索引边界保护。

#### 六、可视化（辅助）
- `script/dataProcess/visualize_results.m`
  - 重建调用改为 `simulation(scatter_all, h, w)`。
  - 频域重建时 `spotlight(..., 84, h, w)`，并为频谱图分配 `zeros(h, w)`。

---

### 兼容性与迁移说明
- 函数签名变更：
  - `simulation(T) -> simulation(T, q_rows, q_cols)`
  - `spotlight(... ) -> spotlight(..., p_in, q_rows, q_cols)`
  - 本仓库中所有调用点已同步更新；若有外部脚本调用，请按新签名修改。
- 默认比例以 128 为基线；若需固定绝对物理尺度，可将比例逻辑替换为基于真实像元间距/视场的计算。

### 建议的验证流程
1) 对同一目标，分别在 128×128 与 158×158 输入上运行提取与重建；比较：
   - 散射中心数量与参数分布（x、y、L、φ''、A、α）。
   - 重建误差（残差范数、PSNR、SSIM）。
2) 在 158×158 场景下对比 p=84 与 p≈104（按比例）：
   - 预期按比例的 p 能更好复现 128 基线的相对行为。

### 后续可选优化
- 读写端字节序统一：`image_read.m` 建议改为 `'b'` 大端读取，完全匹配 `create_R1_for_image_read.m` 的写法。
- 将比例系数（84、30、4）抽取为配置项，便于实验性调参。

### 受影响的文件清单
- 数据处理：
  - `script/dataProcess/step2_MSTAR_mat2raw.m`
  - `script/dataProcess/create_R1_for_image_read.m`
  - `script/dataProcess/image_read.m`（建议项）
- 主流程与重建：
  - `script/dataProcess/step3_main_xulu.m`
  - `script/dataProcess/extrac.m`
  - `script/dataProcess/simulation.m`
  - `script/dataProcess/spotlight.m`
- 检测与分割：
  - `script/dataProcess/TargetDetect.m`
  - `script/dataProcess/watershed_image.m`
- 参数估计：
  - `script/dataProcess/findlocalA.m`
  - `script/dataProcess/finda.m`
  - `script/dataProcess/extraction_local_a0.m`
  - `script/dataProcess/extraction_local_a05.m`
  - `script/dataProcess/extraction_local_a1.m`
  - `script/dataProcess/extraction_local_xy.m`
  - `script/dataProcess/extraction_local_xy_a0.m`
  - `script/dataProcess/extraction_local_xy_a1.m`
  - `script/dataProcess/extraction_dis_a0.m`
- 可视化：
  - `script/dataProcess/visualize_results.m`

---

如需将比例自适配改为“统一固定 p/winsize/半径”的策略，或引入配置文件进行集中管理，可在此日志基础上扩展配置与注释。


