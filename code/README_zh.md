# PEM-Flow 使用说明

视频 → 检测与跟踪 → 背景稳像 → 图像恢复 → 世界坐标投影 → 平滑与运动学 → 自动版 CSV。
`revise` 命令按人工修订规则生成修订版 CSV。

## 安装

使用 Python 3.12，在 `PEM-Flow/code/` 中执行以下命令。

创建虚拟环境：

```shell
python -m venv .venv
```

Windows PowerShell 中激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux 终端中激活环境：

```bash
source .venv/bin/activate
```

选择下面一种安装方式。NVIDIA 显卡及其驱动支持 CUDA 12.8 时：

```shell
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/gpu.txt
python -m pip install -e . --no-deps
```

使用 CPU 时：

```shell
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/cpu.txt
python -m pip install -e . --no-deps
```

后续命令均使用此环境。Windows 下也可将命令中的 `python` 替换为
`.venv\Scripts\python.exe`，直接调用环境中的 Python。
在代码库根目录运行 `git lfs pull` 可下载模型和示例视频。

## 运行示例视频

```shell
python -m pemflow run --workers 2 --device 0 --output ../outputs/demo
```

使用 CPU 时，把 `--device 0` 改为 `--device cpu`。
默认配置为 `configs/demo_dataset.json`，视频位于 `../assets/demos/`。
三段视频各有 300 帧，对应结果位于 `../data/demo/`。每段的帧号和 ID 各自独立，
原视频帧号对应关系见 `../assets/demos/clip_manifest.json`。

加上 `--segments G3-6` 可只处理一个示例。中断后使用相同配置和输出目录，加
`--resume` 接续；单独运行时指定新的输出目录。两个工作进程并行处理两段视频，
每段内部按顺序跟踪。进度显示在终端及输出目录的 `progress.json` 中。

## 使用数据集

[GitHub 仓库](https://github.com/AAA-XIAOJIE/PEM-Flow)提供处理代码、模型、
配置、人工标注 XML、人工修订规则，以及三段示例视频和对应结果。Zenodo 的
`PEMFlow_data.zip` 提供全部实验的正式结果、标注图片、投影输入和下游结果。

解压数据包，将其中 `PEM-Flow/code/` 和 `PEM-Flow/data/` 复制到已下载的
代码库根目录，合并同名目录。合并后，标注图片位于 `code/validation/images/`，
全部实验数据位于 `data/`。以下命令均在 `code/` 中执行。

| 任务 | 命令 |
| --- | --- |
| 应用人工修订 | `python -m pemflow revise --input ../data/automatic --output ../outputs/revised` |
| 计算运动学、密度和流量 | `python -m pemflow metrics --input ../data/revised --output ../outputs/downstream --workers 2` |
| 评价检测和跟踪 | `python -m pemflow evaluate --input ../data/revised --automatic ../data/automatic --output ../outputs/evaluation` |
| 计算检测器 AP | `python examples/score_ap.py --output ../outputs/ap` |

`revise` 使用 `corrections/final_rules.json`，其中的源 ID 和包含首尾的帧范围对应
自动版数据集。AP 命令读取随数据提供的置信度预测记录和人工标注。

利用随数据提供的图像观测和稳像矩阵计算自动版 CSV，再应用人工修订规则：

```shell
python examples/project_observations.py --input ../data/automatic --output ../outputs/automatic
python -m pemflow revise --input ../outputs/automatic --output ../outputs/revised
```

## 处理源视频

通过 `--video-root` 指定输入目录，其中的视频文件名应与数据集配置一致。
处理 `configs/dataset.json` 对应的录像：

```shell
python -m pemflow run --dataset configs/dataset.json --video-root ../videos --output ../outputs/automatic --workers 2 --device 0
```

处理其他录像时，按[配置说明](docs/configuration.md)创建 `configs/my_dataset.json`，
通过 `--dataset` 指定，并填写该录像的相机参数、控制点和标定参考帧。

## 输出

以下目录均相对于命令指定的 `--output`。

| 目录 | 内容 |
| --- | --- |
| `trajectory/` | 位置、速度和相邻位置增量 |
| `motion/` | 速度及加速度分量、连续运动段 |
| `lineage/` | 源观测、身份和逐行来源 |
| `intermediates/` | 视频处理得到的图像观测和稳像矩阵 |

自动版与人工修订版使用相同的 CSV 结构。帧号从 1 开始，ID 仅在各自视频内有效。
G7 两段文件名为 `G3-7.csv`、`G3-7_hmi.csv`。空数值字段表示该值不可用。

详见[字段说明](docs/data_dictionary.md)、[配置说明](docs/configuration.md)和
[许可说明](THIRD_PARTY_NOTICES.md)。运行 `python -m pemflow --help`，或在具体命令
后加 `--help`，可查看该命令的选项。
