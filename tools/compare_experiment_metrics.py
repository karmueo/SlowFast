#!/usr/bin/env python3
"""汇总多个实验的分类指标并按优先级排序。"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 参数对象。
    """
    parser = argparse.ArgumentParser(description="汇总多个 test_metrics.json 并排序")
    parser.add_argument(
        "--metrics",
        type=Path,
        nargs="+",
        required=True,
        help="一个或多个 test_metrics.json 路径",
    )
    parser.add_argument(
        "--groups",
        type=Path,
        nargs="*",
        default=[],
        help="可选，一个或多个 misclassified_groups.json 路径",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="可选，输出汇总 JSON 路径",
    )
    return parser.parse_args()


def load_json(json_path: Path) -> Dict[str, object]:
    """读取 JSON 文件。

    Args:
        json_path (Path): JSON 文件路径。

    Returns:
        Dict[str, object]: 解析后的 JSON 对象。
    """
    return json.loads(json_path.read_text(encoding="utf-8"))


def infer_experiment_name(metrics_path: Path) -> str:
    """从指标文件路径推断实验名。

    Args:
        metrics_path (Path): 指标文件路径。

    Returns:
        str: 实验名。
    """
    return metrics_path.parent.name


def find_group_payload(
    experiment_name: str, group_payloads: Dict[str, Dict[str, object]]
) -> Optional[Dict[str, object]]:
    """按实验名查找对应的误分类分组报告。

    Args:
        experiment_name (str): 实验目录名。
        group_payloads (Dict[str, Dict[str, object]]): 误分类报告映射。

    Returns:
        Optional[Dict[str, object]]: 找到时返回报告，否则返回 None。
    """
    return group_payloads.get(experiment_name)


def summarize_experiment(
    metrics_path: Path, group_payloads: Dict[str, Dict[str, object]]
) -> Dict[str, object]:
    """生成单个实验的摘要。

    Args:
        metrics_path (Path): 指标文件路径。
        group_payloads (Dict[str, Dict[str, object]]): 误分类报告映射。

    Returns:
        Dict[str, object]: 单个实验的摘要信息。
    """
    metrics_payload = load_json(metrics_path)  # 中文注释：当前实验的指标报告。
    experiment_name = infer_experiment_name(metrics_path)  # 中文注释：当前实验目录名。
    classification_report = metrics_payload["classification_report"]  # 中文注释：sklearn 风格的分类报告。
    bird_report = classification_report.get("bird", {})  # 中文注释：bird 类指标字典。
    uav_report = classification_report.get("uav", {})  # 中文注释：uav 类指标字典。
    macro_report = classification_report.get("macro avg", {})  # 中文注释：宏平均指标字典。
    group_payload = find_group_payload(experiment_name, group_payloads)  # 中文注释：与当前实验匹配的误分类分组报告。

    bird_to_uav = 0  # 中文注释：bird->uav 的误分类数量。
    top_groups: List[Dict[str, object]] = []  # 中文注释：高频误分类组摘要。
    wrong_count = None  # 中文注释：总误分类数。
    if group_payload is not None:
        direction_counter = group_payload.get("direction_counter", {})  # 中文注释：误分类方向计数字典。
        bird_to_uav = int(direction_counter.get("bird->uav", 0))
        wrong_count = int(group_payload.get("wrong_count", 0))
        top_groups = [
            {"group": item["group"], "count": int(item["count"])}
            for item in group_payload.get("groups", [])[:3]
        ]

    return {
        "experiment": experiment_name,
        "metrics_path": str(metrics_path),
        "accuracy": float(metrics_payload["accuracy"]),
        "bird_recall": float(bird_report.get("recall", 0.0)),
        "bird_precision": float(bird_report.get("precision", 0.0)),
        "bird_f1": float(bird_report.get("f1-score", 0.0)),
        "uav_recall": float(uav_report.get("recall", 0.0)),
        "uav_precision": float(uav_report.get("precision", 0.0)),
        "uav_f1": float(uav_report.get("f1-score", 0.0)),
        "macro_f1": float(macro_report.get("f1-score", 0.0)),
        "wrong_count": wrong_count,
        "bird_to_uav": bird_to_uav,
        "top_groups": top_groups,
    }


def print_summary_table(experiments: List[Dict[str, object]]) -> None:
    """打印实验排序摘要。

    Args:
        experiments (List[Dict[str, object]]): 已排序的实验摘要列表。
    """
    print(
        "排序规则: bird_recall desc, macro_f1 desc, accuracy desc, bird_to_uav asc"
    )
    header = (
        f"{'experiment':36} {'bird_rec':>8} {'macro_f1':>8} "
        f"{'acc':>8} {'uav_rec':>8} {'b->u':>6} {'wrong':>6}"
    )
    print(header)
    print("-" * len(header))
    for item in experiments:
        wrong_count = "-" if item["wrong_count"] is None else str(item["wrong_count"])  # 中文注释：当前实验总误分类数展示值。
        print(
            f"{item['experiment']:36} "
            f"{item['bird_recall']:.4f} "
            f"{item['macro_f1']:.4f} "
            f"{item['accuracy']:.4f} "
            f"{item['uav_recall']:.4f} "
            f"{item['bird_to_uav']:6d} "
            f"{wrong_count:>6}"
        )
        if item["top_groups"]:
            top_group_text = ", ".join(
                f"{group['group']}({group['count']})" for group in item["top_groups"]
            )  # 中文注释：前三个高频误分类组的摘要字符串。
            print(f"  top_groups: {top_group_text}")


def main() -> None:
    """执行实验汇总流程。"""
    args = parse_args()
    group_payloads: Dict[str, Dict[str, object]] = {}  # 中文注释：实验名到误分类分组报告的映射。
    for group_path in args.groups:
        group_payload = load_json(group_path)  # 中文注释：当前误分类分组 JSON 内容。
        group_payloads[group_path.parent.name] = group_payload

    experiments = [
        summarize_experiment(metrics_path, group_payloads)
        for metrics_path in args.metrics
    ]  # 中文注释：所有实验的摘要列表。
    experiments.sort(
        key=lambda item: (
            item["bird_recall"],
            item["macro_f1"],
            item["accuracy"],
            -item["bird_to_uav"],
        ),
        reverse=True,
    )

    print_summary_table(experiments)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(experiments, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"已写入 JSON 报告: {args.output_json}")


if __name__ == "__main__":
    main()
