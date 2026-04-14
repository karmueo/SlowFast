#!/usr/bin/env python3
"""从 SlowFast 测试结果生成二分类评估报告。"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 参数对象。
    """
    parser = argparse.ArgumentParser(description="生成二分类 precision/recall/F1/confusion matrix 报告")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="TEST.SAVE_RESULTS_PATH 生成的 pickle 文件路径",
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        required=True,
        help="类别映射文件，格式为 '<id> <name>'",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="可选，输出 JSON 报告路径",
    )
    return parser.parse_args()


def load_results(results_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """读取 SlowFast 保存的预测结果。

    Args:
        results_path (Path): pickle 文件路径。

    Returns:
        Tuple[np.ndarray, np.ndarray]: `(preds, labels)`，其中 preds 为 `(N, C)`。
    """
    with open(results_path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 2:
        raise SystemExit(f"无法解析结果文件: {results_path}")

    preds = np.asarray(payload[0])  # 中文注释：模型输出分数矩阵 `(N, C)`。
    labels = np.asarray(payload[1]).reshape(-1)  # 中文注释：样本真实标签向量。
    if preds.ndim != 2:
        raise SystemExit(f"preds 形状异常: {preds.shape}，期望 (N, C)")
    if preds.shape[0] != labels.shape[0]:
        raise SystemExit(
            f"样本数不一致: preds={preds.shape[0]}, labels={labels.shape[0]}"
        )
    return preds, labels


def load_class_map(class_map_path: Path) -> Dict[int, str]:
    """读取类别映射文件。

    Args:
        class_map_path (Path): 映射文件路径。

    Returns:
        Dict[int, str]: `id -> class_name` 的映射。
    """
    id_to_name: Dict[int, str] = {}  # 中文注释：类别编号到类别名的映射。
    with open(class_map_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped_line = raw_line.strip()  # 中文注释：当前去空白后的文本行。
            if not stripped_line:
                continue
            class_id_text, class_name = stripped_line.split(maxsplit=1)  # 中文注释：拆分得到类别编号与类别名。
            id_to_name[int(class_id_text)] = class_name
    if not id_to_name:
        raise SystemExit(f"类别映射为空: {class_map_path}")
    return id_to_name


def build_report(preds: np.ndarray, labels: np.ndarray, id_to_name: Dict[int, str]) -> Dict[str, object]:
    """构建 precision/recall/F1 与混淆矩阵报告。

    Args:
        preds (np.ndarray): 预测分数矩阵。
        labels (np.ndarray): 真实标签。
        id_to_name (Dict[int, str]): 类别映射。

    Returns:
        Dict[str, object]: JSON 可序列化的评估结果。
    """
    label_ids = sorted(id_to_name)  # 中文注释：升序类别编号列表。
    target_names = [id_to_name[label_id] for label_id in label_ids]  # 中文注释：与类别编号对齐的类别名列表。
    pred_labels = preds.argmax(axis=1)  # 中文注释：top1 预测标签。
    probabilities = preds.astype(np.float64)  # 中文注释：用于导出每条样本概率的分数矩阵。
    if probabilities.shape[1] == 2:
        probabilities = probabilities - probabilities.max(axis=1, keepdims=True)
        probabilities = np.exp(probabilities)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    report_dict = classification_report(
        labels,
        pred_labels,
        labels=label_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, pred_labels, labels=label_ids)  # 中文注释：按类别顺序生成的混淆矩阵。
    accuracy = float((pred_labels == labels).mean())  # 中文注释：整体准确率。

    return {
        "accuracy": accuracy,
        "class_map": {str(label_id): id_to_name[label_id] for label_id in label_ids},
        "confusion_matrix": matrix.tolist(),
        "classification_report": report_dict,
        "sample_count": int(labels.shape[0]),
        "probabilities": probabilities.tolist(),
        "pred_labels": pred_labels.tolist(),
        "labels": labels.tolist(),
    }


def main() -> None:
    """执行评估报告生成流程。"""
    args = parse_args()
    preds, labels = load_results(args.results)
    id_to_name = load_class_map(args.class_map)
    report = build_report(preds, labels, id_to_name)

    print(f"样本数: {report['sample_count']}")
    print(f"准确率: {report['accuracy'] * 100:.2f}%")
    print("混淆矩阵:")
    for row in report["confusion_matrix"]:
        print("  " + " ".join(str(value) for value in row))
    print("分类报告:")
    for class_name, metrics_dict in report["classification_report"].items():
        if not isinstance(metrics_dict, dict):
            continue
        precision = metrics_dict.get("precision", 0.0)  # 中文注释：当前类别或聚合项的 precision。
        recall = metrics_dict.get("recall", 0.0)  # 中文注释：当前类别或聚合项的 recall。
        f1_score = metrics_dict.get("f1-score", 0.0)  # 中文注释：当前类别或聚合项的 F1。
        support = metrics_dict.get("support", 0)  # 中文注释：当前类别或聚合项的样本数。
        print(
            f"  {class_name}: precision={precision:.4f}, recall={recall:.4f}, "
            f"f1={f1_score:.4f}, support={support}"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"已写入 JSON 报告: {args.output_json}")


if __name__ == "__main__":
    main()
