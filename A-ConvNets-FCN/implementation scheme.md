# 一、目标与产出（总述）

 **目标** ：在你现有的 five-channel 标注体系上，复现论文中的  **ASC 参数估计网络** （A-ConvNets + FCN 聚合），其 **输出仅为 2 个通道** （A 与 α 的稀疏图）；位置 (x,y)(x,y) 通过 **A 通道的峰值**相对图像中心读出。

 **产出** ：一个独立子模块 `A-ConvNets-FCN/`，包含数据转换、模型、训练、推理与评测脚本，能与现有仓库 **并行对比** 。

 **关键一致性** ：

* **标签** ：从 5 通道（heat, dx, dy, A, alpha）**转换出** 2 通道稀疏图（A_map、alpha_map）。
* **损失** ：论文范式 —— `全图 MSE + 非零区 MSE`（对 A 与 α 分别计算后求和）。
* **结构** ：收缩（8/16/32/64, 5×5 Conv+BN+ReLU+2×2 Pool）→ FCN 式多尺度上采样（从 4×4、8×8、16×16直接上采样到 128×128 **逐分支上采样并相加** ）→ 1×1 卷积输出 2 通道。

---

# 二、实现步骤（分述）

## 0) 目录结构（放在仓库根目录）

```
A-ConvNets-FCN/
  ├── README.md
  ├── config.yaml
  ├── convert_labels.py        # 5→2 通道离线转换（或训练时 on-the-fly）
  ├── dataset.py               # 数据集与增强/归一化
  ├── model.py                 # A-ConvNets + FCN 聚合
  ├── losses.py                # 全图 MSE + 非零区 MSE
  ├── train.py                 # 训练主脚本
  ├── infer.py                 # 推理与峰值读数
  ├── eval.py                  # 评测（定位+参数回归）
  ├── utils/
  │    ├── peaks.py            # NMS/局部极大值检测
  │    ├── seed.py             # 随机种子/确定性
  │    └── io.py               # 读写/日志/可视化
  └── outputs/                 # 运行产出（ckpt、日志、可视化）
```

## 1) 数据与输入

 **输入影像** ：与你当前仓库一致（单通道幅度或 2 通道实部/虚部都可；（保持与主仓库一致即可）。

 **标签转换（5→2）** ：参考script\step4_label_preprocess.py对mat数据的读取方式，但是最后要生成2通道的label，**通道0：A_map（推荐线性幅度 A）** ，**通道1：alpha_map（α）**。每个散射中心仅在**一个**像素写值（四舍五入后的峰值像素），其他像素为 0。目录结构与现有输出一致（保留相对子目录），便于并行对比训练。位置 (x,y)(x,y)**(**x**,**y**)** 不单独作为标签图通道；推理时通过 **A_map 的峰值坐标**来读出位置并换算成“相对图心”的 (x,y)(x,y)**(**x**,**y**)**。这与论文读数方式一致。

## 2) 模型结构（A-ConvNets + FCN 聚合）

**收缩（Contracting）**

* `ConvBlock(k=5)`：Conv2d → BN → ReLU
* 通道：`[1→8] → [8→16] → [16→32] → [32→64]`
* 每层后 `MaxPool2d(2)`，空间分辨率：`128→64→32→16→8`，再加一层使最深可到 `4×4`（可在第4层后再加一次池化）。

**FCN 聚合（Multi-branch）**

* 取末三层特征图：`F4(4×4, C=64)`, `F3(8×8, C=32)`, `F2(16×16, C=16)`
* 用 `Conv1x1` 统一到同一通道数（如 16），分别用 `ConvTranspose2d` 上采样到 `128×128`（比例 ×32, ×16, ×8）
* 把三路上采样结果逐像素相加，接 `Conv1x1` 输出  **2 通道** （`A`, `alpha`）。

 **伪代码（网络骨架）** ：

```python
# model.py
class ConvBlk:
    def __init__(in_c, out_c, k=5, s=1, p=2):
        Conv2d(in_c, out_c, k, s, p)
        BN(out_c)
        ReLU()

class AConvFCN(nn.Module):
    def __init__(self, out_channels=2):
        # contracting
        c1 = ConvBlk(1, 8);   pool1 = MaxPool2d(2)   # 128->64
        c2 = ConvBlk(8,16);   pool2 = MaxPool2d(2)   # 64->32
        c3 = ConvBlk(16,32);  pool3 = MaxPool2d(2)   # 32->16
        c4 = ConvBlk(32,64);  pool4 = MaxPool2d(2)   # 16->8
        c5 = ConvBlk(64,64);  pool5 = MaxPool2d(2)   # 8->4

        # fcn heads (1x1 reduce + deconv to 128x128)
        head5 = Conv1x1(64,16); up5 = Deconv(to=128)  # scale x32
        head4 = Conv1x1(64,16); up4 = Deconv(to=128)  # scale x16 from 8x8
        head3 = Conv1x1(32,16); up3 = Deconv(to=128)  # scale x8  from 16x16

        out = Conv1x1(16, out_channels)  # final 2ch

    def forward(x):
        x1 = c1(x); x = pool1(x1)
        x2 = c2(x); x = pool2(x2)
        x3 = c3(x); x = pool3(x3)
        x4 = c4(x); x = pool4(x4)
        x5 = c5(x); x = pool5(x5)

        y5 = up5(head5(x5))  # -> 128x128
        y4 = up4(head4(x4))  # -> 128x128
        y3 = up3(head3(x3))  # -> 128x128

        y = y5 + y4 + y3
        return out(y)        # (B,2,128,128)
```

> 注：上采样可用 `ConvTranspose2d(kernel=4, stride=2, padding=1)` 级联若干次，或一次性 `F.interpolate(..., size=128, mode='bilinear')` 后接卷积，二者任选其一，保持**多尺度相加**即可。

## 3) 损失函数（论文范式）

 **定义** ：对 A 与 α 两通道分别计算

* `MSE_full = mse(pred, gt)`
* `MSE_nz = mse(pred[gt!=0], gt[gt!=0])`（若无非零像素则置 0）

  **总损失** ：`Loss = (MSE_full_A + MSE_nz_A) + (MSE_full_alpha + MSE_nz_alpha)`

 **伪代码** ：

```python
# losses.py
def asc_loss(pred, gt):  # (B,2,H,W)
    total = 0
    for ch in [0,1]:
        p, g = pred[:,ch], gt[:,ch]
        nz = (g != 0)
        loss_full = mse(p, g)
        loss_nz   = mse(p[nz], g[nz]) if nz.any() else 0.0
        total += loss_full + loss_nz
    return total
```

## 4) 训练流程

 **超参建议** ：

* Optimizer: Adam(lr=1e-3), Cosine/StepLR
* Batch: 32（按显存调整）
* Epochs: 100
* Aug: 轻量随机翻转/旋转 90°，不要破坏稀疏标签（同步变换 A/α 图）

 **伪代码（训练主循环）** ：

```python
# train.py
set_seed(42)
model = AConvFCN(out_channels=2).to(device)
opt = Adam(model.parameters(), lr=1e-3)
for epoch in range(epochs):
    model.train()
    for img, label5 in loader:
        # on-the-fly 或 预先保存的2通道
        A_alpha = five_to_two(label5)        # (2,H,W)
        pred = model(img)                    # (B,2,H,W)
        loss = asc_loss(pred, A_alpha)
        opt.zero_grad(); loss.backward(); opt.step()
    # 验证：计算 loss 与 “非零位置”的 MAE
    save_ckpt_if_best()
```

## 5) 推理与读数

 **单目标图** ：

* 从 `pred[:,0]`（A 通道）取全局最大值坐标 `(r*,c*)`；
* 读 `A_hat = pred[0, r*, c*]`, `alpha_hat = pred[1, r*, c*]`；
* 位置相对中心：`x_hat = c* - W//2`, `y_hat = r* - H//2`。

 **多目标图（可选）** ：

* 用 NMS 找 Top-K 局部峰（如 K≤3）；在同位置读 `(A, α)`；
* 与 GT 峰集合做匈牙利匹配，评估位置/参数误差。

 **伪代码** ：

```python
# infer.py
pred = model(img)          # (1,2,128,128)
A_map = pred[0,0]; alpha_map = pred[0,1]
r,c = argmax2d(A_map)
A_hat, alpha_hat = A_map[r,c], alpha_map[r,c]
x_hat, y_hat = c - 64, r - 64
return (x_hat, y_hat, A_hat, alpha_hat)
```
