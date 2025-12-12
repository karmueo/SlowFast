import argparse
import re
from pathlib import Path
from typing import Dict, List


def load_class_map(path: Path) -> Dict[str, int]:
    """
    读取类别映射文件，格式: "<id> <folder>" 或 "<id>,<folder>"，支持 # 注释。
    返回 {folder: id}
    """
    mapping: Dict[str, int] = {}
    with open(path) as f:
        for ln in f:
            ln = ln.split("#", 1)[0].strip()
            if not ln:
                continue
            parts = [p for p in re.split(r"[,\s]+", ln) if p]
            if len(parts) != 2:
                raise SystemExit(f"格式错误: {ln}")
            idx, name = parts
            mapping[name] = int(idx)
    if not mapping:
        raise SystemExit(f"映射文件为空: {path}")
    return mapping


def rewrite_csv(csv_path: Path, class_map: Dict[str, int], sep: str, backup: bool):
    """
    按 class_map 覆写 CSV 中的类别 ID，格式: frame_dir<num_frames><label(int)>。
    仅修改第三列 label；前两列保持不变。
    """
    with open(csv_path) as f:
        lines = f.readlines()

    rewritten: List[str] = []
    for ln in lines:
        ln_stripped = ln.strip()
        if not ln_stripped:
            continue
        parts = ln_stripped.split(sep)
        if len(parts) < 3:
            raise SystemExit(f"{csv_path}: 行格式错误，少于3列 -> {ln_stripped}")
        frame_dir, num_frames, _ = parts[:3]
        # 类别目录为 frame_dir 的第一个路径段
        class_name = frame_dir.split("/")[0]
        if class_name not in class_map:
            raise SystemExit(f"{csv_path}: 未在映射中找到类别 {class_name}")
        parts[2] = str(class_map[class_name])
        rewritten.append(sep.join(parts))

    if backup:
        csv_path.with_suffix(csv_path.suffix + ".bak").write_text("".join(lines))
    csv_path.write_text("\n".join(rewritten) + ("\n" if rewritten else ""))
    print(f"重写完成: {csv_path} ({len(rewritten)} 行)")


def main():
    ap = argparse.ArgumentParser("Rewrite CSV labels using class map file")
    ap.add_argument("--csv-dir", type=Path, required=True, help="包含 train/val/test.csv 的目录")
    ap.add_argument("--class-map", type=Path, required=True, help="映射文件: '<id> <folder>' 每行")
    ap.add_argument("--separator", type=str, default=" ", help="分隔符，需与原 CSV 一致")
    ap.add_argument("--no-backup", action="store_true", help="不写入 .bak 备份")
    args = ap.parse_args()

    csv_dir = args.csv_dir
    class_map = load_class_map(args.class_map)
    sep = args.separator
    backup = not args.no_backup

    targets = [p for p in csv_dir.glob("*.csv") if p.is_file()]
    if not targets:
        raise SystemExit(f"未找到 CSV 文件: {csv_dir}")

    for csv_path in targets:
        rewrite_csv(csv_path, class_map, sep, backup)


if __name__ == "__main__":
    main()
