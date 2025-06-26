"""
OMP ASC提取器配置文件
===================

定义不同场景下的OMP算法配置参数，包括精度优先、速度优先等模式。
"""

import numpy as np

# ============================================================================
# 基础配置
# ============================================================================

# SAR成像系统参数
SAR_PARAMS = {
    "fc": 1e10,  # 中心频率 10 GHz
    "B": 5e8,  # 带宽 500 MHz
    "om_deg": 2.86,  # 俯仰角度
    "p": 84,  # K空间采样点数
    "q": 128,  # 图像尺寸
    "c": 3e8,  # 光速
}

# 图像参数
IMAGE_PARAMS = {
    "img_size": (128, 128),
    "pixel_spacing": 0.1,  # 像素间距 (米)
}

# ============================================================================
# OMP配置方案
# ============================================================================

OMP_CONFIGS = {
    # 高精度模式 - 用于研究和精确分析
    "high_precision": {
        "search_region": (-6.4, 6.4, -6.4, 6.4),  # 搜索区域 (米)
        "grid_resolution": 0.1,  # 精细网格 (米)
        "alpha_range": (-2.0, 2.0),  # Alpha参数范围
        "alpha_steps": 21,  # Alpha离散化步数
        "n_nonzero_coefs": 25,  # 最大散射中心数
        "cross_validation": True,  # 启用交叉验证
        "cv_folds": 5,  # 交叉验证折数
        "max_iter": 100,  # 最大迭代次数
        "tol": 1e-6,  # 收敛阈值
        "normalize_y": True,  # 归一化观测向量
        "precompute": "auto",  # 预计算Gram矩阵
        "description": "高精度模式：精细网格，详细参数搜索，适合研究分析",
    },
    # 标准模式 - 平衡精度和速度
    "standard": {
        "search_region": (-6.4, 6.4, -6.4, 6.4),
        "grid_resolution": 0.2,  # 标准网格
        "alpha_range": (-1.5, 1.5),
        "alpha_steps": 11,
        "n_nonzero_coefs": 20,
        "cross_validation": False,
        "cv_folds": 3,
        "max_iter": 50,
        "tol": 1e-5,
        "normalize_y": True,
        "precompute": "auto",
        "description": "标准模式：平衡精度和计算效率，适合常规处理",
    },
    # 快速模式 - 优先考虑计算速度
    "fast": {
        "search_region": (-6.0, 6.0, -6.0, 6.0),  # 略小的搜索区域
        "grid_resolution": 0.4,  # 粗糙网格
        "alpha_range": (-1.0, 1.0),
        "alpha_steps": 7,
        "n_nonzero_coefs": 15,
        "cross_validation": False,
        "cv_folds": 3,
        "max_iter": 30,
        "tol": 1e-4,
        "normalize_y": True,
        "precompute": True,  # 强制预计算以加速
        "description": "快速模式：粗糙网格，快速处理，适合批量分析",
    },
    # 稀疏模式 - 检测少量强散射中心
    "sparse": {
        "search_region": (-6.4, 6.4, -6.4, 6.4),
        "grid_resolution": 0.15,
        "alpha_range": (-2.0, 2.0),
        "alpha_steps": 15,
        "n_nonzero_coefs": 10,  # 强制稀疏
        "cross_validation": False,
        "cv_folds": 3,
        "max_iter": 40,
        "tol": 1e-5,
        "normalize_y": True,
        "precompute": "auto",
        "description": "稀疏模式：检测少量主要散射中心，避免过拟合",
    },
    # 密集模式 - 检测更多散射中心
    "dense": {
        "search_region": (-6.4, 6.4, -6.4, 6.4),
        "grid_resolution": 0.1,
        "alpha_range": (-2.0, 2.0),
        "alpha_steps": 17,
        "n_nonzero_coefs": 35,  # 允许更多散射中心
        "cross_validation": True,
        "cv_folds": 3,
        "max_iter": 80,
        "tol": 1e-6,
        "normalize_y": True,
        "precompute": "auto",
        "description": "密集模式：检测更多散射中心，捕获细节特征",
    },
    # 调试模式 - 最小配置用于快速测试
    "debug": {
        "search_region": (-3.0, 3.0, -3.0, 3.0),  # 小搜索区域
        "grid_resolution": 0.8,  # 很粗的网格
        "alpha_range": (-0.5, 0.5),
        "alpha_steps": 3,
        "n_nonzero_coefs": 5,
        "cross_validation": False,
        "cv_folds": 2,
        "max_iter": 10,
        "tol": 1e-3,
        "normalize_y": False,
        "precompute": False,
        "description": "调试模式：最小配置，快速验证算法流程",
    },
}

# ============================================================================
# 后处理配置
# ============================================================================

POSTPROCESS_CONFIGS = {
    # 散射中心过滤参数
    "filtering": {
        "min_amplitude_ratio": 0.01,  # 最小幅度比例（相对于最强散射中心）
        "min_absolute_amplitude": 1e-3,  # 最小绝对幅度
        "spatial_clustering_threshold": 0.2,  # 空间聚类阈值 (米)
        "merge_close_scatterers": True,  # 是否合并邻近散射中心
    },
    # 可视化参数
    "visualization": {
        "figsize": (15, 12),
        "dpi": 300,
        "colormap": "gray",
        "marker_size": 12,
        "marker_color": "red",
        "marker_style": "+",
        "font_size": 12,
        "save_format": "png",
    },
    # 评估指标
    "evaluation": {
        "rmse_threshold": 0.3,  # RMSE阈值
        "correlation_threshold": 0.7,  # 相关系数阈值
        "snr_calculation": True,  # 是否计算SNR
        "entropy_calculation": True,  # 是否计算熵值
    },
}

# ============================================================================
# 辅助函数
# ============================================================================


def get_config(config_name="standard"):
    """
    获取指定的配置方案

    Parameters:
    -----------
    config_name : str
        配置名称，可选: 'high_precision', 'standard', 'fast', 'sparse', 'dense', 'debug'

    Returns:
    --------
    dict
        包含完整配置参数的字典
    """
    if config_name not in OMP_CONFIGS:
        available_configs = list(OMP_CONFIGS.keys())
        raise ValueError(f"未知配置名称: {config_name}. 可用配置: {available_configs}")

    # 合并所有配置
    config = {
        "sar_params": SAR_PARAMS.copy(),
        "image_params": IMAGE_PARAMS.copy(),
        "omp_params": OMP_CONFIGS[config_name].copy(),
        "postprocess": POSTPROCESS_CONFIGS.copy(),
        "config_name": config_name,
    }

    return config


def list_available_configs():
    """列出所有可用的配置方案"""
    print("🔧 可用的OMP配置方案:")
    print("=" * 60)
    for name, config in OMP_CONFIGS.items():
        grid_size = config["grid_resolution"]
        n_coefs = config["n_nonzero_coefs"]
        cv = "是" if config["cross_validation"] else "否"

        print(f"📋 {name:15s} - {config['description']}")
        print(f"    网格分辨率: {grid_size}m, 最大散射中心: {n_coefs}, 交叉验证: {cv}")
        print()


def estimate_computation_cost(config_name="standard"):
    """估算指定配置的计算复杂度"""
    config = get_config(config_name)
    omp_params = config["omp_params"]

    # 计算字典大小
    search_region = omp_params["search_region"]
    grid_res = omp_params["grid_resolution"]

    x_size = int((search_region[1] - search_region[0]) / grid_res) + 1
    y_size = int((search_region[3] - search_region[2]) / grid_res) + 1
    alpha_steps = omp_params["alpha_steps"]

    n_atoms = x_size * y_size * alpha_steps
    n_measurements = SAR_PARAMS["p"] ** 2

    # 估算内存使用（复数字典转实数）
    memory_mb = (n_measurements * n_atoms * 2 * 8) / (1024**2)  # 双精度实数

    # 估算计算时间（经验公式）
    relative_time = (n_atoms / 10000) * (omp_params["n_nonzero_coefs"] / 20)

    print(f"💾 配置 '{config_name}' 的计算复杂度估算:")
    print(f"   字典尺寸: {n_measurements} x {n_atoms}")
    print(f"   内存需求: ~{memory_mb:.1f} MB")
    print(f"   相对计算时间: {relative_time:.2f}x (相对于标准配置)")

    return {
        "n_atoms": n_atoms,
        "n_measurements": n_measurements,
        "memory_mb": memory_mb,
        "relative_time": relative_time,
    }


# ============================================================================
# 配置验证
# ============================================================================


def validate_config(config):
    """验证配置参数的合理性"""
    omp_params = config["omp_params"]

    warnings = []

    # 检查网格分辨率
    if omp_params["grid_resolution"] < 0.05:
        warnings.append("⚠️ 网格分辨率过小，可能导致计算量过大")

    if omp_params["grid_resolution"] > 1.0:
        warnings.append("⚠️ 网格分辨率过大，可能影响定位精度")

    # 检查稀疏度
    search_region = omp_params["search_region"]
    grid_res = omp_params["grid_resolution"]
    x_size = int((search_region[1] - search_region[0]) / grid_res) + 1
    y_size = int((search_region[3] - search_region[2]) / grid_res) + 1
    n_positions = x_size * y_size

    if omp_params["n_nonzero_coefs"] > n_positions * 0.1:
        warnings.append("⚠️ 稀疏度设置可能过高，容易过拟合")

    # 检查Alpha范围
    if abs(omp_params["alpha_range"][1] - omp_params["alpha_range"][0]) > 4:
        warnings.append("⚠️ Alpha参数范围过大，建议限制在[-2, 2]内")

    if warnings:
        print("配置验证警告:")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("✅ 配置参数验证通过")

    return len(warnings) == 0


if __name__ == "__main__":
    # 演示配置系统
    print("🎯 OMP ASC提取器配置系统演示")
    print("=" * 50)

    # 列出所有配置
    list_available_configs()

    # 演示配置获取和验证
    for config_name in ["debug", "fast", "standard", "high_precision"]:
        print(f"\n📊 配置方案: {config_name}")
        print("-" * 30)

        config = get_config(config_name)
        validate_config(config)
        estimate_computation_cost(config_name)
        print()
