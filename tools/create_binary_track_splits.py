#!/usr/bin/env python3
"""生成按源序列分组的二分类 train/val/test 划分。"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_CLASS_TO_ID = {"bird": 0, "uav": 1}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 命令行参数对象。
    """
    parser = argparse.ArgumentParser(description="生成按源序列安全切分的 uav/bird 二分类 CSV")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="包含类别目录的根目录，例如 bird/uav/plane 所在目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出 train.csv/val.csv/test.csv 的目录",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["bird", "uav"],
        help="参与训练的类别目录名，默认 bird uav",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="训练集比例，默认 0.7",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="验证集比例，默认 0.15",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="测试集比例，默认 0.15",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子，默认 0",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=1,
        help="样本最少帧数，默认 1",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default=" ",
        help="CSV 列分隔符，默认空格",
    )
    return parser.parse_args()


def count_frames(sample_dir: Path) -> int:
    """统计样本目录中的图片帧数。

    Args:
        sample_dir (Path): 样本目录路径。

    Returns:
        int: 图片帧数量。
    """
    frame_count = 0  # 中文注释：记录当前样本目录中的帧数。
    for image_path in sample_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            frame_count += 1
    return frame_count


def infer_group_key(sample_path: Path) -> str:
    """从样本目录名推断源序列分组键。

    约定样本名形如 `Smart_Record_xxx_7`，最后一段为同源轨迹内的切片编号。

    Args:
        sample_path (Path): 相对根目录的样本路径，例如 `bird/foo_3`。

    Returns:
        str: 组键，保证同源样本落在同一个 split。
    """
    class_name = sample_path.parts[0]  # 中文注释：类别目录名，用于避免跨类别同名冲突。
    sample_name = sample_path.parts[-1]  # 中文注释：样本目录名。
    matched = re.match(r"(.+)_\d+$", sample_name)  # 中文注释：匹配结尾切片编号。
    base_name = matched.group(1) if matched else sample_name  # 中文注释：去掉切片编号后的源序列名。
    return f"{class_name}/{base_name}"


def collect_grouped_samples(
    root_dir: Path,
    class_names: Sequence[str],
    min_frames: int,
) -> Tuple[Dict[str, List[Tuple[str, int, int]]], Dict[str, int]]:
    """收集并按源序列分组样本。

    Args:
        root_dir (Path): 数据根目录。
        class_names (Sequence[str]): 保留的类别目录名列表。
        min_frames (int): 最小帧数要求。

    Returns:
        Tuple[Dict[str, List[Tuple[str, int, int]]], Dict[str, int]]:
            - 分组样本字典，键为组键，值为样本列表 `(相对路径, 帧数, 标签)`。
            - 类别到整数标签的映射。
    """
    grouped_samples: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)  # 中文注释：保存按源序列聚合后的样本。
    class_to_id = {name: DEFAULT_CLASS_TO_ID[name] for name in class_names}  # 中文注释：当前二分类标签映射。

    for class_name in class_names:
        class_dir = root_dir / class_name  # 中文注释：当前类别目录。
        if not class_dir.is_dir():
            raise SystemExit(f"未找到类别目录: {class_dir}")
        class_id = class_to_id[class_name]  # 中文注释：当前类别的整数标签。
        for sample_dir in sorted(class_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            frame_count = count_frames(sample_dir)  # 中文注释：当前样本帧数。
            if frame_count < min_frames:
                continue
            relative_path = sample_dir.relative_to(root_dir).as_posix()  # 中文注释：写入 CSV 的相对路径。
            group_key = infer_group_key(Path(relative_path))  # 中文注释：当前样本对应的源序列分组键。
            grouped_samples[group_key].append((relative_path, frame_count, class_id))

    if not grouped_samples:
        raise SystemExit("未收集到任何满足条件的样本，请检查路径或 min-frames 设置。")
    return grouped_samples, class_to_id


def split_group_keys(
    group_keys: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Dict[str, List[str]]:
    """将一类样本的组键按比例划分到 train/val/test。

    Args:
        group_keys (Sequence[str]): 当前类别的所有组键。
        train_ratio (float): 训练集比例。
        val_ratio (float): 验证集比例。
        test_ratio (float): 测试集比例。
        rng (random.Random): 随机数生成器。

    Returns:
        Dict[str, List[str]]: split 名称到组键列表的映射。
    """
    shuffled_keys = list(group_keys)  # 中文注释：可被随机打乱的组键副本。
    rng.shuffle(shuffled_keys)
    total_groups = len(shuffled_keys)  # 中文注释：当前类别的分组总数。

    if total_groups < 3:
        raise SystemExit(
            f"类别分组数过少，无法稳定切成 train/val/test: {total_groups}"
        )

    train_count = max(1, int(round(total_groups * train_ratio)))  # 中文注释：训练集分组数。
    val_count = max(1, int(round(total_groups * val_ratio)))  # 中文注释：验证集分组数。
    if train_count + val_count >= total_groups:
        val_count = max(1, total_groups - train_count - 1)
    test_count = total_groups - train_count - val_count  # 中文注释：测试集分组数。
    if test_count <= 0:
        test_count = 1
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1

    train_keys = shuffled_keys[:train_count]  # 中文注释：训练集组键。
    val_keys = shuffled_keys[train_count : train_count + val_count]  # 中文注释：验证集组键。
    test_keys = shuffled_keys[train_count + val_count :]  # 中文注释：测试集组键。

    if not train_keys or not val_keys or not test_keys:
        raise SystemExit(
            "切分结果存在空 split，请调整比例或补充数据。"
        )

    return {"train": train_keys, "val": val_keys, "test": test_keys}


def build_split_rows(
    grouped_samples: Dict[str, List[Tuple[str, int, int]]],
    class_names: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Tuple[str, int, int]]]:
    """按类别独立切分组键，最终汇总为 CSV 行。

    Args:
        grouped_samples (Dict[str, List[Tuple[str, int, int]]]): 分组后的样本字典。
        class_names (Sequence[str]): 参与训练的类别名列表。
        train_ratio (float): 训练集比例。
        val_ratio (float): 验证集比例。
        test_ratio (float): 测试集比例。
        seed (int): 随机种子。

    Returns:
        Dict[str, List[Tuple[str, int, int]]]: split 到样本行列表的映射。
    """
    rng = random.Random(seed)  # 中文注释：固定随机种子以保证可复现。
    split_rows: Dict[str, List[Tuple[str, int, int]]] = {"train": [], "val": [], "test": []}  # 中文注释：最终输出的三套样本行。
    class_group_keys: Dict[str, List[str]] = defaultdict(list)  # 中文注释：每个类别对应的组键列表。

    for group_key in grouped_samples:
        class_name = group_key.split("/", 1)[0]  # 中文注释：当前组键所属类别名。
        if class_name in class_names:
            class_group_keys[class_name].append(group_key)

    for class_name in class_names:
        current_group_keys = class_group_keys[class_name]  # 中文注释：当前类别的全部组键。
        split_keys = split_group_keys(
            current_group_keys,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            rng=rng,
        )
        for split_name, keys in split_keys.items():
            for group_key in keys:
                split_rows[split_name].extend(grouped_samples[group_key])

    for split_name in split_rows:
        split_rows[split_name].sort(key=lambda row: row[0])
    return split_rows


def write_csv(csv_path: Path, rows: Iterable[Tuple[str, int, int]], separator: str) -> int:
    """写出 CSV 文件。

    Args:
        csv_path (Path): 输出文件路径。
        rows (Iterable[Tuple[str, int, int]]): 样本行列表。
        separator (str): 字段分隔符。

    Returns:
        int: 写出的样本数量。
    """
    row_list = list(rows)  # 中文注释：物化迭代器以便统计条数。
    with open(csv_path, "w", encoding="utf-8") as handle:
        for relative_path, frame_count, class_id in row_list:
            handle.write(
                f"{relative_path}{separator}{frame_count}{separator}{class_id}\n"
            )
    return len(row_list)


def build_summary(
    split_rows: Dict[str, List[Tuple[str, int, int]]],
    grouped_samples: Dict[str, List[Tuple[str, int, int]]],
) -> Dict[str, object]:
    """构建切分摘要信息。

    Args:
        split_rows (Dict[str, List[Tuple[str, int, int]]]): split 到样本行的映射。
        grouped_samples (Dict[str, List[Tuple[str, int, int]]]): 原始分组样本字典。

    Returns:
        Dict[str, object]: 便于保存为 JSON 的摘要结构。
    """
    group_to_split: Dict[str, str] = {}  # 中文注释：记录每个组最终落入的 split。
    split_stats: Dict[str, Dict[str, int]] = {}  # 中文注释：统计每个 split 的样本数与类别数。

    for split_name, rows in split_rows.items():
        class_counter: Dict[str, int] = defaultdict(int)  # 中文注释：当前 split 的类别样本计数。
        for relative_path, _frame_count, class_id in rows:
            class_name = Path(relative_path).parts[0]  # 中文注释：从相对路径恢复类别名。
            class_counter[class_name] += 1
            group_to_split[infer_group_key(Path(relative_path))] = split_name
        split_stats[split_name] = {
            "samples": len(rows),
            **{f"class_{name}": count for name, count in sorted(class_counter.items())},
        }

    leakage_groups = []  # 中文注释：记录异常跨 split 的组键，正常情况下应为空。
    for group_key in grouped_samples:
        recorded_split = group_to_split.get(group_key)  # 中文注释：当前组被记录的 split。
        if recorded_split is None:
            leakage_groups.append(group_key)

    return {
        "split_stats": split_stats,
        "group_count": len(grouped_samples),
        "unassigned_groups": leakage_groups,
    }


def main() -> None:
    """执行按源序列切分的主流程。"""
    args = parse_args()
    root_dir = args.root.resolve()  # 中文注释：标准化后的数据根目录。
    output_dir = args.output_dir.resolve()  # 中文注释：标准化后的输出目录。
    class_names = list(args.classes)  # 中文注释：本次参与训练的类别列表。

    if set(class_names) - set(DEFAULT_CLASS_TO_ID):
        unsupported = sorted(set(class_names) - set(DEFAULT_CLASS_TO_ID))  # 中文注释：当前脚本不支持的类别列表。
        raise SystemExit(f"仅支持这些类别: {sorted(DEFAULT_CLASS_TO_ID)}，收到: {unsupported}")

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio  # 中文注释：三套比例之和。
    if ratio_sum <= 0:
        raise SystemExit("train/val/test 比例之和必须大于 0。")

    grouped_samples, class_to_id = collect_grouped_samples(
        root_dir=root_dir,
        class_names=class_names,
        min_frames=args.min_frames,
    )
    split_rows = build_split_rows(
        grouped_samples=grouped_samples,
        class_names=class_names,
        train_ratio=args.train_ratio / ratio_sum,
        val_ratio=args.val_ratio / ratio_sum,
        test_ratio=args.test_ratio / ratio_sum,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_counts = {}  # 中文注释：各 CSV 实际写出条数。
    for split_name, rows in split_rows.items():
        csv_path = output_dir / f"{split_name}.csv"  # 中文注释：当前 split 的输出 CSV 路径。
        write_counts[split_name] = write_csv(csv_path, rows, args.separator)

    class_map_path = output_dir / "class_map.txt"  # 中文注释：类别映射文本文件路径。
    with open(class_map_path, "w", encoding="utf-8") as handle:
        for class_name, class_id in sorted(class_to_id.items(), key=lambda item: item[1]):
            handle.write(f"{class_id} {class_name}\n")

    summary = build_summary(split_rows, grouped_samples)  # 中文注释：切分摘要信息。
    summary.update(
        {
            "root_dir": str(root_dir),
            "output_dir": str(output_dir),
            "class_to_id": class_to_id,
            "write_counts": write_counts,
            "seed": args.seed,
        }
    )
    summary_path = output_dir / "split_summary.json"  # 中文注释：摘要 JSON 路径。
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"切分完成: {output_dir}")
    for split_name in ("train", "val", "test"):
        print(f"  {split_name}: {write_counts[split_name]}")
    print(f"类别映射: {class_map_path}")
    print(f"摘要文件: {summary_path}")


if __name__ == "__main__":
    main()
