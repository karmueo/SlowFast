#!/usr/bin/env python3
"""按源序列统计 SlowFast 的误分类分布。"""

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 命令行参数对象。
    """
    parser = argparse.ArgumentParser(description="按源序列聚合误分类样本")
    parser.add_argument("--results", type=Path, required=True, help="test_results.pkl 路径")
    parser.add_argument("--data-dir", type=Path, required=True, help="包含 test.csv 的 split 目录")
    parser.add_argument("--class-map", type=Path, required=True, help="类别映射文件 class_map.txt")
    parser.add_argument("--csv-name", type=str, default="test.csv", help="测试集 CSV 文件名")
    parser.add_argument("--sep", type=str, default=" ", help="CSV 字段分隔符")
    parser.add_argument("--topn", type=int, default=20, help="最多打印多少个高频误分类组")
    parser.add_argument("--out-json", type=Path, default=None, help="可选，输出 JSON 路径")
    parser.add_argument("--out-csv", type=Path, default=None, help="可选，输出 CSV 路径")
    return parser.parse_args()


def load_results(results_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """读取预测结果文件。

    Args:
        results_path (Path): `test_results.pkl` 路径。

    Returns:
        Tuple[np.ndarray, np.ndarray]: `(preds, labels)`。
    """
    with open(results_path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 2:
        raise SystemExit(f"无法解析结果文件: {results_path}")
    preds = np.asarray(payload[0])  # 中文注释：模型输出分数矩阵。
    labels = np.asarray(payload[1]).reshape(-1)  # 中文注释：真实标签向量。
    return preds, labels


def load_rows(csv_path: Path, separator: str) -> List[Tuple[str, int]]:
    """读取测试集 CSV 中的相对路径与标签。

    Args:
        csv_path (Path): 测试 CSV 路径。
        separator (str): 字段分隔符。

    Returns:
        List[Tuple[str, int]]: `(relative_path, label)` 列表。
    """
    rows: List[Tuple[str, int]] = []  # 中文注释：顺序与测试结果对齐的样本行列表。
    with open(csv_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()  # 中文注释：去掉空白后的当前文本行。
            if not stripped:
                continue
            parts = stripped.split(separator)  # 中文注释：按分隔符拆分后的字段列表。
            rows.append((parts[0], int(parts[-1])))
    return rows


def load_class_map(class_map_path: Path) -> Dict[int, str]:
    """读取类别映射文件。

    Args:
        class_map_path (Path): `class_map.txt` 路径。

    Returns:
        Dict[int, str]: 类别编号到类别名的映射。
    """
    id_to_name: Dict[int, str] = {}  # 中文注释：类别编号到类别名的映射表。
    with open(class_map_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()  # 中文注释：去掉空白后的当前文本行。
            if not stripped:
                continue
            class_id_text, class_name = stripped.split(maxsplit=1)  # 中文注释：当前类别编号与类别名。
            id_to_name[int(class_id_text)] = class_name
    return id_to_name


def infer_group_key(relative_path: str) -> str:
    """从样本相对路径恢复源序列组键。

    Args:
        relative_path (str): 形如 `bird/foo_3` 的相对路径。

    Returns:
        str: 去掉切片编号后的组键。
    """
    class_name, sample_name = relative_path.split("/", 1)  # 中文注释：类别名与样本目录名。
    group_name = sample_name.rsplit("_", 1)[0]  # 中文注释：去掉切片编号后的源序列名。
    return f"{class_name}/{group_name}"


def build_group_report(
    preds: np.ndarray,
    labels: np.ndarray,
    rows: List[Tuple[str, int]],
    id_to_name: Dict[int, str],
) -> Dict[str, object]:
    """构建按源序列统计的误分类报告。

    Args:
        preds (np.ndarray): 预测分数矩阵。
        labels (np.ndarray): 真实标签向量。
        rows (List[Tuple[str, int]]): 测试 CSV 行。
        id_to_name (Dict[int, str]): 类别映射。

    Returns:
        Dict[str, object]: 可序列化的统计报告。
    """
    pred_labels = preds.argmax(axis=1)  # 中文注释：top1 预测标签。
    group_counter: Counter = Counter()  # 中文注释：每个源序列的误分类次数。
    direction_counter: Counter = Counter()  # 中文注释：每种误分类方向的次数。
    group_details: Dict[str, List[Dict[str, object]]] = defaultdict(list)  # 中文注释：每个误分类组下的样本详情。

    for index, ((relative_path, gt_label), pred_label) in enumerate(zip(rows, pred_labels)):
        pred_label = int(pred_label)
        if int(gt_label) == pred_label:
            continue
        group_key = infer_group_key(relative_path)  # 中文注释：当前样本所属源序列键。
        group_counter[group_key] += 1
        direction_key = f"{id_to_name[int(gt_label)]}->{id_to_name[pred_label]}"  # 中文注释：当前误分类方向字符串。
        direction_counter[direction_key] += 1
        group_details[group_key].append(
            {
                "index": index,
                "relative_path": relative_path,
                "gt": id_to_name[int(gt_label)],
                "pred": id_to_name[pred_label],
                "confidence": float(preds[index].max()),
            }
        )

    sorted_groups = [  # 中文注释：按误分类数量降序排列后的组统计。
        {
            "group": group_key,
            "count": count,
            "details": group_details[group_key],
        }
        for group_key, count in group_counter.most_common()
    ]
    return {
        "wrong_count": int(sum(group_counter.values())),
        "direction_counter": dict(direction_counter),
        "groups": sorted_groups,
    }


def main() -> None:
    """执行按源序列聚合误分类的主流程。"""
    args = parse_args()
    preds, labels = load_results(args.results)
    rows = load_rows(args.data_dir / args.csv_name, args.sep)
    id_to_name = load_class_map(args.class_map)

    if len(rows) != preds.shape[0]:
        raise SystemExit(
            f"测试 CSV 与结果条数不一致: rows={len(rows)} preds={preds.shape[0]}"
        )

    report = build_group_report(preds, labels, rows, id_to_name)
    print(f"总误分类数: {report['wrong_count']}")
    print("误分类方向统计:")
    for direction, count in report["direction_counter"].items():
        print(f"  {direction}: {count}")
    print("高频误分类组:")
    for item in report["groups"][: args.topn]:
        print(f"  {item['group']}: {item['count']}")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"已写入 JSON 报告: {args.out_json}")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["group", "count", "index", "relative_path", "gt", "pred", "confidence"],
            )
            writer.writeheader()
            for item in report["groups"]:
                for detail in item["details"]:
                    writer.writerow(
                        {
                            "group": item["group"],
                            "count": item["count"],
                            **detail,
                        }
                    )
        print(f"已写入 CSV 报告: {args.out_csv}")


if __name__ == "__main__":
    main()
