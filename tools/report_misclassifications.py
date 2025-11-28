#!/usr/bin/env python3
import argparse
import os
import pickle
import csv
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np


def load_results(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    # Expect [all_preds, all_labels]
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        preds, labels = obj
    else:
        raise SystemExit(
            f"无法解析结果文件: {pkl_path}. 期望 [preds, labels] 的 pickle。"
        )
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    if preds.ndim != 2:
        raise SystemExit(f"preds 形状异常: {preds.shape}, 期望 (N, C)")
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if preds.shape[0] != labels.shape[0]:
        raise SystemExit(
            f"样本数不一致: preds={preds.shape[0]}, labels={labels.shape[0]}"
        )
    return preds, labels


def read_csv_list(csv_path: Path, sep: str = " ") -> List[Tuple[str, Optional[int]]]:
    rows = []
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # kinetics.py 支持两列或三列，我们兼容：
            parts = line.split(sep)
            if len(parts) == 1:
                # 单列：路径，label=0
                rows.append((parts[0], 0))
            elif len(parts) == 2:
                # 两列：路径 标签
                rows.append((parts[0], int(parts[1])))
            else:
                # 三列：路径 帧数 标签
                rows.append((parts[0], int(parts[-1])))
    return rows


def build_label_map_from_root(root: Path) -> List[str]:
    # 与 create_csv_files.py 的规则一致：按类别目录名排序
    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    names = sorted([p.name for p in class_dirs])
    return names  # index 对应 label id


def topk_from_scores(scores: np.ndarray, k: int):
    idx = np.argsort(-scores)[:k]
    return idx, scores[idx]


def main():
    ap = argparse.ArgumentParser("Report misclassified samples from SlowFast test_results.pkl")
    ap.add_argument("--results", type=Path, required=True, help="Path to test_results.pkl saved by TEST.SAVE_RESULTS_PATH")
    ap.add_argument("--data_dir", type=Path, required=True, help="DATA.PATH_TO_DATA_DIR (contains test.csv)")
    ap.add_argument("--path_prefix", type=Path, required=True, help="DATA.PATH_PREFIX (root of frame dirs)")
    ap.add_argument("--csv_name", type=str, default="test.csv", help="CSV filename, default: test.csv")
    ap.add_argument("--sep", type=str, default=" ", help="CSV field separator, default: space")
    ap.add_argument("--topk", type=int, default=5, help="Show top-k predictions for each mistake")
    ap.add_argument("--out_csv", type=Path, default=None, help="Optional path to save a CSV report of misclassifications")
    args = ap.parse_args()

    preds, labels = load_results(args.results)
    csv_path = args.data_dir / args.csv_name
    if not csv_path.exists():
        raise SystemExit(f"找不到CSV文件: {csv_path}")
    rows = read_csv_list(csv_path, sep=args.sep)

    num_videos = len(rows)
    if num_videos != preds.shape[0]:
        print(
            f"警告: 结果条数与CSV不一致。preds={preds.shape[0]} vs csv rows={num_videos}.\n"
            "可能原因：多视角设置或CSV不匹配。仍将按 min(N) 对齐。"
        )
    n = min(num_videos, preds.shape[0])
    rows = rows[:n]
    preds = preds[:n]
    labels = labels[:n]

    # 尝试还原 label -> name
    label_names = None
    try:
        label_names = build_label_map_from_root(args.path_prefix)
    except Exception:
        label_names = None

    pred_top1 = preds.argmax(axis=1)
    correct = pred_top1 == labels
    idx_wrong = np.where(~correct)[0].tolist()

    print(f"总样本: {n}, 正确: {correct.sum()} ({correct.mean()*100:.2f}%), 错误: {len(idx_wrong)}")

    report_rows = []
    for i in idx_wrong:
        rel_path, gt = rows[i]
        scores = preds[i]
        topk_idx, topk_scores = topk_from_scores(scores, min(args.topk, scores.shape[0]))
        def label_str(x: int) -> str:
            if label_names and 0 <= x < len(label_names):
                return f"{x}:{label_names[x]}"
            return str(x)
        topk_str = ", ".join([f"{label_str(int(ci))}={float(cs):.4f}" for ci, cs in zip(topk_idx, topk_scores)])
        row = {
            "index": i,
            "rel_path": rel_path,
            "abs_path": str((args.path_prefix / rel_path).resolve()),
            "gt": int(gt),
            "gt_name": label_names[int(gt)] if (label_names and 0 <= int(gt) < len(label_names)) else "",
            "pred": int(pred_top1[i]),
            "pred_name": label_names[int(pred_top1[i])] if (label_names and 0 <= int(pred_top1[i]) < len(label_names)) else "",
            "topk": topk_str,
        }
        report_rows.append(row)

    # 打印前若干条
    if report_rows:
        print("\n误分类样本 (最多显示前 50 条):")
        for r in report_rows[:50]:
            print(
                f"[{r['index']}] {r['rel_path']} | gt={r['gt']} {r['gt_name']} -> pred={r['pred']} {r['pred_name']} | top{args.topk}: {r['topk']}"
            )
    else:
        print("没有发现误分类样本。")

    # 可选保存为CSV
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index",
                    "rel_path",
                    "abs_path",
                    "gt",
                    "gt_name",
                    "pred",
                    "pred_name",
                    "topk",
                ],
            )
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"\n已将误分类报告写入: {args.out_csv}")


if __name__ == "__main__":
    main()
