# PEM-Flow 使用说明

本代码处理十八组实验的十九段视频。正式流程为：

**视频 → YOLO11m 检测及 ByteTrack 跟踪 → 背景稳像与逐帧质量检查 → 图像恢复短时共同框 → 世界坐标投影 → 连续轨迹运动学 → CSV。**

人工审阅产生明确的修订记录，`revise` 执行这些记录。`metrics` 计算轨迹、密度和流量指标，`evaluate` 计算相对于人工 GT 的检测与跟踪质量。

## 安装与输入

已测试的环境为 Python 3.12，Torch 2.9.1 + CUDA 12.8、Ultralytics 8.3.235、OpenCV 4.12.0.88、NumPy 2.2.6、pandas 2.2.3、SciPy 1.15.3、Shapely 2.1.1。

在 `code/` 中执行：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements/gpu.txt
.venv\Scripts\python -m pip install -e . --no-deps
```

安装文件提供已测试的环境组合。程序在输出中记录实际环境，使用当前安装的依赖运行。选择与本机兼容的 PyTorch CPU 或 CUDA 安装包；CPU 推理使用 `--device cpu`。

也可以为 CPU 计算单独创建环境，安装 `requirements/cpu.txt` 和 `pip install -e . --no-deps`。完整运行入口支持用 `--inference-python`、`--cpu-python` 指定两个解释器。

原视频放在 `../assets/videos/`，文件名见 `../assets/video_manifest.csv`。train12 权重已放在 `../assets/model/train12/best.pt`。相机参数、控制点、参考帧、稳像和恢复参数见 `configs/dataset.json`；ROI、网格和目标形状参数见 `configs/pipeline.json`。

G7 两段分别是：

| 处理键 | 视频 | CSV |
| --- | --- | --- |
| `G3-7-1` | `G3-7.mp4` | `G3-7.csv` |
| `G3-7-2` | `h_G3-7-1.mp4` | `G3-7_hmi.csv` |

两个视频独立处理帧号与身份，实验统计合并为 G07。

## 四个入口

```powershell
.venv\Scripts\python -m pemflow run --workers 2 --device 0 --output ../outputs/automatic
.venv\Scripts\python -m pemflow revise --input ../outputs/automatic --output ../outputs/revised
.venv\Scripts\python -m pemflow metrics --input ../outputs/revised --output ../outputs/downstream --workers 2
.venv\Scripts\python -m pemflow evaluate --input ../outputs/revised --automatic ../outputs/automatic --output ../outputs/evaluation
```

以上命令在 `code/` 中执行，直接使用安装时创建的环境。激活该环境后，也可以写成 `pemflow ...` 或 `python -m pemflow ...`。默认输入位置以发布包为基准；显式传入的相对路径以执行命令的目录为基准。

两个工作进程可使用同一 GPU。每个视频内部仍按顺序逐帧计算，批量大小为 1，跟踪状态互相独立。终端显示视频、阶段、帧数和已完成视频数量，输出目录保留 `progress.json` 与阶段日志。

中断后重复原命令并加 `--resume`。已完成阶段直接保留；未完成的跟踪从该视频开头重算，稳像的已完成阶段可接续。新的输入或科学参数使用新的输出目录。

处理一部分视频时指定处理键，例如：

```powershell
.venv\Scripts\python -m pemflow run --segments G3-1 G3-7-2 --workers 2 --output ../outputs/selected
```

## 修订执行

`corrections/final_rules.json` 是最终的人工判断记录，共 77 条。每条写明原始来源 ID、视频帧范围、操作、原因与人工证据。

删除操作按名单和范围过滤行；身份映射同时赋值；用户提交的完整框用生产矩阵投影。涉及位置或身份连续性的轨迹，用同一套投影、平滑和导数函数重算。其余字段保留原 CSV 文本。审计记录包含修改前后值、行来源和触发规则。

这份规则对应随包保存的自动版。若换用不同的原始跟踪结果，应重新审阅规则和身份对应。程序中的范围及行数检查用于确认规则选中的数据，不能替代人工判断。

## 指标与质量评价

`metrics` 读取 CSV、运动学表和行谱系，产生轨迹合理性统计、ROI 个体表和逐帧密度/流量表。保留真实缺失导数；ROI 无人的帧密度和流量为零，ROI 有目标但速度缺失的帧流量保留缺失。

轨迹汇总包含速度、SG 切向加速度绝对值、SG 加速度模长和连续段位置增量。运动学表中的切向加速度保留正负号，汇总其绝对值用于合理性统计；加速度模长也包含改变方向的影响。

`evaluate` 使用固定的 270 帧检测 GT 和 600 帧跟踪 GT，计算 P/R/F1、MOTA、IDF1、IDSW、Frag 和同相机投影框中心差异。GT 用于评价，视频生成使用视频、模型和科学参数。

检测器 AP 需要视频中的原尺寸图像，在推理环境执行：

```powershell
.venv\Scripts\python -m pemflow evaluate --ap --input ../data/revised --automatic ../data/automatic --output ../outputs/evaluation
```

该入口解码 270 个固定采样帧，以置信度 0.001 重新检测，使用 IoU 0.50–0.95 和 101 点精度包络计算 AP。

## 结果

`../data/automatic/` 有 7,143,147 行，`../data/revised/` 有 6,887,738 行，两版均对应 132,663 个源视频帧。修订版 ROI 有 2,820,289 行。两版轨迹图与 CSV 一起保存，图中采用统一世界坐标范围和英文标注。

CSV 的列为 `frame,id,class,real_x,real_y,velocity,dist`。帧号从 1 开始，坐标和距离单位为米，速度为 m/s；类别为 0 行人、9 电动自行车、3 实验机动车。详细定义见 [methods.md](docs/methods.md) 和 [字段说明](docs/data_dictionary.md)。
