import argparse
import os
import sys
from pathlib import Path

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from dataset import five_to_two  # noqa: E402
from utils.io import ensure_dir, load_config  # noqa: E402

"""
功能
----
将 ASC 数据集中的 5 通道标签 (5-ch) 批量转换为 2 通道标签 (2-ch)，
以便 A-ConvNets-FCN 模型使用。转换逻辑由 `dataset.five_to_two` 提供。

输入/输出
--------
- 输入根目录: 在配置文件 `config.yaml` 的 `data.label_5ch_root` 指定；
  程序会递归遍历其下的所有 `.npy` 文件作为 5-ch 标签输入。
- 输出根目录: 在配置文件 `config.yaml` 的 `data.label_2ch_root` 指定；
  程序将按与输入相同的子目录结构，将对应的 2-ch 标签以 `.npy` 保存。

行为说明
--------
- 对每个 `.npy` 文件:
  1) 加载为 NumPy 数组；
  2) 调用 `five_to_two(label_5ch)` 完成 5→2 通道映射；
  3) 将结果保存到输出目录（若已存在将被覆盖）。
- 若 `five_to_two` 由于数据格式不符合预期而抛出 `ValueError`，该样本会被跳过并计数。
- 运行结束后，会打印“已转换/已跳过”的文件数量统计。

配置项
------
- `data.label_5ch_root`: 5 通道标签的根目录（递归处理）。
- `data.label_2ch_root`: 2 通道标签的输出根目录（自动创建子目录）。

使用示例
--------
命令行:
  python convert_labels.py --config config.yaml

或在项目根目录中（示例路径按实际调整）:
  python A-ConvNets-FCN/convert_labels.py --config config.yaml

备注
----
- 本脚本仅负责批量 IO 与流程控制；具体的通道语义与形状要求由
  `dataset.five_to_two` 决定，请确保输入标签满足其预期格式。
"""


def convert_labels(config_path: str) -> None:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    label_5ch_root = Path(data_cfg["label_5ch_root"])
    label_2ch_root = Path(data_cfg["label_2ch_root"])
    ensure_dir(str(label_2ch_root))

    converted = 0
    skipped = 0

    for dirpath, _, filenames in os.walk(label_5ch_root):
        for filename in filenames:
            if not filename.endswith(".npy"):
                continue
            src_path = Path(dirpath) / filename
            rel_dir = Path(dirpath).relative_to(label_5ch_root)
            dst_dir = label_2ch_root / rel_dir
            ensure_dir(str(dst_dir))
            dst_path = dst_dir / filename

            label_5ch = np.load(src_path)
            try:
                label_2ch = five_to_two(label_5ch)
            except ValueError:
                skipped += 1
                continue
            np.save(dst_path, label_2ch)
            converted += 1

    print(f"Converted {converted} label files. Skipped {skipped} files.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert 5-channel ASC labels to 2-channel (A, alpha).")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    convert_labels(args.config)


if __name__ == "__main__":
    main()
