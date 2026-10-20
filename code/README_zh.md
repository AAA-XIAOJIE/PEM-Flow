# PEM-Flow 使用说明

处理流程为：

**视频 → YOLO11m检测与ByteTrack跟踪 → 背景稳像及图像检查 → 图像恢复 → 世界坐标投影 → 运动学计算 → CSV。**

人工审阅形成修订记录，`revise`执行记录；`metrics`计算运动、密度和流量，`evaluate`计算检测与跟踪质量。

## 安装

使用Python 3.11及以上版本，已测试环境为Python 3.12。在`code/`中执行：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements/gpu.txt
.venv\Scripts\python -m pip install -e . --no-deps
```

按本机设备选择PyTorch安装包；CPU推理使用`--device cpu`，CPU依赖组合见`requirements/cpu.txt`。程序在输出中记录实际依赖版本。

将Zenodo数据包中的`PEM-Flow/`合并到代码库同名目录。train12权重位于`../assets/model/train12/best.pt`。

## 运行示例视频

三段10秒视频位于`../assets/demos/`，分别对应24、48、72名参与者及50%电动自行车占比：

```powershell
.venv\Scripts\python -m pemflow run --dataset configs/demo_dataset.json --video-root ../assets/demos --workers 2 --device 0 --output ../outputs/demo
```

对应的轨迹、运动学和行来源结果位于`../data/demo/`。每个片段独立跟踪，帧号从1开始。源视频帧范围与编码信息见`assets/demos/clip_manifest.json`。相机标定保持对应原视频的参数；G15标定参考帧为片段内第151帧，即索引150。

## 重算数据集

使用`../data/automatic/`中的观测和稳像矩阵，调用正式投影与运动学函数重算19段轨迹：

```powershell
.venv\Scripts\python examples/reproject_cached.py --output ../outputs/reprojected
.venv\Scripts\python -m pemflow revise --input ../outputs/reprojected --output ../outputs/revised
.venv\Scripts\python -m pemflow metrics --input ../outputs/revised --output ../outputs/downstream --workers 2
.venv\Scripts\python -m pemflow evaluate --input ../outputs/revised --automatic ../outputs/reprojected --output ../outputs/evaluation
```

对应结果保存在`../data/automatic/`、`../data/revised/`及`../data/downstream/`。自动轨迹共7,143,147行，人工审阅后的轨迹共6,887,738行，覆盖132,663个源视频帧；ROI共2,819,547行。

从270帧低阈值预测记录及人工标注重算检测AP：

```powershell
.venv\Scripts\python examples/score_archived_ap.py --output ../outputs/ap
```

重算参考线试验的高度比较：

```powershell
.venv\Scripts\python examples/reproduce_height_selection.py --output ../outputs/height_selection
```

投影高度为行人0.72 m、电动自行车0.20 m、机动车0.60 m。前两类各采用两次往返、共四个单程，以运动学处理后单程RMSE的算术平均值选高，平均横向偏差分别为4.02225 cm、4.16203 cm。高度试验使用自身的相机标定及道路高程，输入见`../data/height_selection/`。

## 处理原视频

将`../assets/video_manifest.csv`列出的19个视频放到`../assets/videos/`。清单给出文件名、大小和SHA-256。

```powershell
.venv\Scripts\python -m pemflow run --workers 2 --device 0 --output ../outputs/automatic
.venv\Scripts\python -m pemflow revise --input ../outputs/automatic --output ../outputs/revised
.venv\Scripts\python -m pemflow metrics --input ../outputs/revised --output ../outputs/downstream --workers 2
.venv\Scripts\python -m pemflow evaluate --input ../outputs/revised --automatic ../outputs/automatic --output ../outputs/evaluation
```

相机参数、控制点、参考帧及处理参数位于`configs/dataset.json`；ROI与目标形状参数位于`configs/pipeline.json`。

| 处理键 | 视频 | 轨迹CSV |
| --- | --- | --- |
| `G3-7-1` | `G3-7.mp4` | `G3-7.csv` |
| `G3-7-2` | `h_G3-7-1.mp4` | `G3-7_hmi.csv` |

两段视频的帧号和身份独立，实验统计合并为G07。

两个工作进程可共用一块GPU，每段视频顺序解码，批量大小为1，跟踪状态独立。终端和`progress.json`记录进度。中断后在原命令后加`--resume`接续；用`--segments G3-1 G3-7-2`指定部分视频。`--inference-python`与`--cpu-python`可指定独立运行环境。

上述命令在`code/`中执行；默认输入以代码包位置为基准，显式相对路径以当前工作目录为基准。

## 人工修订与评价

`corrections/final_rules.json`保存77条人工判断，使用`data/automatic/`中的源ID和包含首尾的帧范围。删除操作过滤对应行，ID映射同时赋值，提交的框用正式矩阵投影。受影响的完整轨迹采用同一运动学方法重算，逐行记录改动与原因。

`metrics`生成运动合理性统计及ROI逐个体、逐帧密度和流量。`evaluate`使用270帧检测标注和600帧跟踪标注，计算P/R/F1、MOTA、IDF1、IDSW、Frag及投影框中心差异。

从原视频重新检测并计算AP：

```powershell
.venv\Scripts\python -m pemflow evaluate --ap --input ../data/revised --automatic ../data/automatic --output ../outputs/evaluation
```

该入口解码270个采样帧，以置信度0.001检测，用IoU 0.50–0.95及101点精度包络计算AP。

## 输出与代码结构

自动结果包含`trajectory/`、`motion/`、`lineage/`、`intermediates/`、`dataset.json`、`run.json`及进度日志。轨迹字段为：

```text
frame,id,class,real_x,real_y,velocity,dist
```

位置和距离单位为米，速度为m/s；类别0为行人、9为电动自行车、3为机动车。位置、速度、加速度使用二阶Savitzky–Golay拟合，最大窗口31帧；25–30帧连续段采用合法的最大奇数窗口，更短的段保留观测位置和缺失导数。

`src/pemflow/pipeline/`负责视频处理，`revision/`执行人工记录，`downstream/`计算运动和ROI指标，`evaluation/`进行标注评价。`cli.py`提供命令入口，`execution.py`调度任务，`io.py`处理数据路径。

详细定义见[方法说明](docs/methods.md)、[字段说明](docs/data_dictionary.md)和[许可说明](THIRD_PARTY_NOTICES.md)。
