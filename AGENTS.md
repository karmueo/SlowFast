# Repository Guidelines

## 项目结构与模块
- 核心库在 `slowfast/`（配置、数据集、模型、工具函数）；新增模块需遵守现有包结构。
- 实验配置位于 `configs/`（数据集/模型 YAML）和 `projects/`（专项 README 与配置）。
- 训练/评估入口在 `tools/`（`run_net.py` 负责 train/test，含性能与推理辅助）；演示与可视化资源在 `demo/` 与 `VISUALIZATION_TOOLS.md`。
- 转换/推理工具集中在脚本旁：`export_onnx_final.py`、`tensorrt_inference.py`、`inspect_onnx.py`、`cpp_inference/` 等。

## 构建、测试与开发命令
- 项目使用conda虚拟环境 conda activate SlowFast，已安装相关依赖。
- 可编辑安装：`python setup.py build develop`。
- 保证导入路径：`export PYTHONPATH=$(pwd)/slowfast:$PYTHONPATH`（建议写入 shell 配置）。
- 训练/评估：`python tools/run_net.py --cfg configs/Kinetics/C2D_8x8_R50.yaml DATA.PATH_TO_DATA_DIR path/to/data NUM_GPUS 1 TRAIN.BATCH_SIZE 8`；仅测试时设 `TRAIN.ENABLE False` 并传 `TEST.CHECKPOINT_FILE_PATH`。
- 快速冒烟（CPU/单卡）：追加 `DATA_LOADER.NUM_WORKERS 0 NUM_GPUS 1 TRAIN.BATCH_SIZE 4` 缩短迭代。
- 代码规范检查：`bash linter.sh`（依次运行 isort、black `-l 80`、flake8；若存在则运行 `arc lint`）。

## 代码风格与命名
- Python 使用 4 空格缩进，目标 80 列；保持 PEP8/flake8 通过。isort 100 列换行且分组见 `setup.cfg`。
- 配置遵循既有 YAML 键名（如 `DATA.PATH_TO_DATA_DIR`、`TRAIN.BATCH_SIZE`），数据集/模型名与目录保持一致（如 `configs/Kinetics/X3D_...`）。
- 文件/模块用 lowercase_with_underscores；类用 CapWords；函数与变量用 lowercase_with_underscores。

## 测试指引
- 以小批量/低线程的短训或评估验证改动，尤其是模型头或路径修改后确认 checkpoint 仍可加载。
- 转换流程（ONNX/TensorRT）请用示例输入 `input_data.npz`/`python_input.npy` 运行 `export_onnx_final.py`、`tensorrt_inference.py`、`inspect_onnx.py`，核对输出一致性。
- 新增工具旁尽量补充断言或轻量测试；能设定随机种子时保持确定性。

## 提交与 PR 要求
- 提交信息清晰、动作导向，可参照历史的编号式摘要（如：`更新 X3D 配置：调整 num classes；刷新输出路径`）。
- 从 `master` 建分支，保持 PR 聚焦；说明改动内容、原因与验证方式（命令/日志）。关联 Issue 时请附链接并注明影响的数据集或配置。
- 在发起审阅前运行 `bash linter.sh`，并执行至少一个代表性的 `tools/run_net.py`（或相关转换/推理脚本）。如有速度/精度影响务必写明。

## 回复说明
回复时用中文回答。
