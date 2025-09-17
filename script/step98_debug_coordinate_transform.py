"""
脚本用途: 坐标变换单元测试工具
- 验证 `model_to_pixel` 与 `pixel_to_model` 两个方向的坐标变换是否一致可逆。
- 打印典型点的正向/逆向转换结果与误差，并对图像中心点进行边界测试。

适用场景:
- 调试/确认物理坐标与像素坐标的转换公式、比例系数与中心定义是否正确。
"""

# debug_coordinate_transform.py
import numpy as np


# --- 从你的项目中复制这两个函数 ---
def model_to_pixel(x_model, y_model):
    IMG_HEIGHT, IMG_WIDTH, P_GRID_SIZE = 128, 128, 84
    C1 = IMG_HEIGHT / (0.3 * P_GRID_SIZE)
    C2 = IMG_WIDTH / 2 + 1
    col_pixel = (x_model * C1) + C2
    row_pixel = C2 - (y_model * C1)
    return row_pixel, col_pixel


def pixel_to_model(row, col):
    IMG_HEIGHT, IMG_WIDTH, P_GRID_SIZE = 128, 128, 84
    C1 = IMG_HEIGHT / (0.3 * P_GRID_SIZE)
    C2 = IMG_WIDTH / 2 + 1
    y_model = (C2 - row) / C1
    x_model = (col - C2) / C1
    return x_model, y_model


# --- 进行单元测试 ---
# 选取一个典型的物理坐标
test_x_model, test_y_model = 1.5, -2.0

# 转换为像素坐标
r_pixel, c_pixel = model_to_pixel(test_x_model, test_y_model)
print(f"Model ({test_x_model:.4f}, {test_y_model:.4f}) -> Pixel ({r_pixel:.4f}, {c_pixel:.4f})")

# 将像素坐标转换回物理坐标
x_reconstructed, y_reconstructed = pixel_to_model(r_pixel, c_pixel)
print(f"Pixel ({r_pixel:.4f}, {c_pixel:.4f}) -> Model ({x_reconstructed:.4f}, {y_reconstructed:.4f})")

# 检查误差
error = np.sqrt((test_x_model - x_reconstructed) ** 2 + (test_y_model - y_reconstructed) ** 2)
print(f"\nReconstruction Error: {error}")
assert error < 1e-9, "Coordinate transform is NOT correctly reversible!"
print("\n✅ Coordinate transform test passed!")

# --- 测试边界情况 ---
# 图像中心 (64.5, 64.5) 应该对应物理坐标 (0, 0)
x_center, y_center = pixel_to_model(64.5, 64.5)
print(f"Pixel center (64.5, 64.5) -> Model ({x_center:.4f}, {y_center:.4f})")
assert abs(x_center) < 1e-9 and abs(y_center) < 1e-9

print("✅ Center point test passed!")
