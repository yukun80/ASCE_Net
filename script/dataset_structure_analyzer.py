#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集路径结构分析程序
功能：递归遍历指定目录，以树形结构输出所有文件夹（忽略文件）
作者：ASCE_Net项目
日期：2024
"""

import os
import sys
from pathlib import Path


def print_directory_tree(directory_path, prefix="", max_depth=None, current_depth=0, ignore_patterns=None):
    """
    递归打印目录树结构（仅文件夹）

    Args:
        directory_path (str): 要分析的目录路径
        prefix (str): 当前行的前缀字符
        max_depth (int, optional): 最大递归深度
        current_depth (int): 当前递归深度
        ignore_patterns (list, optional): 要忽略的目录名模式
    """
    if ignore_patterns is None:
        ignore_patterns = [".git", "__pycache__", ".vscode", ".idea", "node_modules"]

    if max_depth is not None and current_depth >= max_depth:
        return

    try:
        directory = Path(directory_path)
        if not directory.exists():
            print(f"错误：目录 {directory_path} 不存在")
            return

        if not directory.is_dir():
            print(f"错误：{directory_path} 不是一个目录")
            return

        # 获取所有子目录（忽略文件）
        subdirs = []
        try:
            for item in directory.iterdir():
                if item.is_dir() and item.name not in ignore_patterns:
                    subdirs.append(item)
        except PermissionError:
            print(f"{prefix}├── [权限被拒绝]")
            return

        # 按名称排序
        subdirs.sort(key=lambda x: x.name.lower())

        # 打印每个子目录
        for i, subdir in enumerate(subdirs):
            is_last = i == len(subdirs) - 1

            # 选择合适的树形字符
            if is_last:
                current_prefix = "└── "
                next_prefix = prefix + "    "
            else:
                current_prefix = "├── "
                next_prefix = prefix + "│   "

            # 统计子目录数量
            try:
                subdir_count = len([x for x in subdir.iterdir() if x.is_dir()])
                if subdir_count > 0:
                    print(f"{prefix}{current_prefix}{subdir.name}/ ({subdir_count} subdirs)")
                else:
                    print(f"{prefix}{current_prefix}{subdir.name}/")
            except PermissionError:
                print(f"{prefix}{current_prefix}{subdir.name}/ [权限被拒绝]")
                continue

            # 递归处理子目录
            print_directory_tree(str(subdir), next_prefix, max_depth, current_depth + 1, ignore_patterns)

    except Exception as e:
        print(f"遍历目录时出错：{e}")


def analyze_dataset_structure(base_path, max_depth=None):
    """
    分析数据集结构的主函数

    Args:
        base_path (str): 数据集根目录路径
        max_depth (int, optional): 最大递归深度
    """
    print("=" * 80)
    print("数据集路径结构分析")
    print("=" * 80)
    print(f"分析目录：{base_path}")
    print(f"最大深度：{'无限制' if max_depth is None else max_depth}")
    print("-" * 80)

    # 打印根目录
    base_dir = Path(base_path)
    if base_dir.exists():
        print(f"{base_dir.name}/")
        print_directory_tree(base_path, "", max_depth)
    else:
        print(f"错误：目录 {base_path} 不存在")

    print("-" * 80)
    print("分析完成")


def get_directory_stats(directory_path):
    """
    获取目录统计信息

    Args:
        directory_path (str): 目录路径

    Returns:
        dict: 包含统计信息的字典
    """
    stats = {"total_dirs": 0, "total_files": 0, "depth_levels": 0}

    def count_recursive(path, current_depth=0):
        try:
            for item in Path(path).iterdir():
                if item.is_dir():
                    stats["total_dirs"] += 1
                    stats["depth_levels"] = max(stats["depth_levels"], current_depth + 1)
                    count_recursive(item, current_depth + 1)
                else:
                    stats["total_files"] += 1
        except PermissionError:
            pass

    count_recursive(directory_path)
    return stats


def main():
    """主函数"""
    # 默认数据集路径
    default_dataset_path = r"E:\Document\paper_library\3rd_paper_250512\code\ASCE_Net\datasets\SAR_ASC_Project"

    # 检查命令行参数
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = default_dataset_path

    # 检查是否指定最大深度
    max_depth = None
    if len(sys.argv) > 2:
        try:
            max_depth = int(sys.argv[2])
        except ValueError:
            print("警告：最大深度参数无效，将使用无限制深度")

    # 如果使用相对路径，转换为当前工作目录下的路径
    if not os.path.isabs(dataset_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        dataset_path = os.path.join(project_root, dataset_path)

    print(f"使用数据集路径：{dataset_path}")

    # 执行分析
    analyze_dataset_structure(dataset_path, max_depth)

    # 显示统计信息
    if os.path.exists(dataset_path):
        stats = get_directory_stats(dataset_path)
        print("\n统计信息：")
        print(f"  总文件夹数：{stats['total_dirs']}")
        print(f"  总文件数：{stats['total_files']}")
        print(f"  最大深度：{stats['depth_levels']}")


if __name__ == "__main__":
    main()
