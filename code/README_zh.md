# PEM-Flow 使用说明

视频 → 检测与跟踪 → 背景稳像 → 图像恢复 → 世界坐标投影 → 运动学计算 → CSV。
人工修订规则形成核验后的数据；指标与评价命令读取相应数据集。

## 安装

使用 Python 3.11 及以上版本，已测试环境为 Python 3.12。在 `code/` 中执行：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements/gpu.txt
.venv\Scripts\python -m pip install -e . --no-deps
```

按设备选择 PyTorch 安装包。CPU 使用 `requirements/cpu.txt` 和 `--device cpu`。
程序在运行记录中保存实际依赖版本。模型和示例视频通过 Git LFS 下载；已有克隆若只有
指针文件，运行 `git lfs pull`。

## 运行示例视频

```powershell
.venv\Scripts\python -m pemflow run --workers 2 --device 0 --output ../outputs/demo
```

默认配置为 `configs/demo_dataset.json`，视频位于 `../assets/demos/`。
三段视频均为 300 帧、3840 × 2160、30 fps，直接复制原视频码流截取。
对应的轨迹、运动学和行来源结果保存在 `../data/demo/`。片段使用独立的帧号和跟踪 ID；
源帧对应关系见 `../assets/demos/clip_manifest.json`。

两个工作进程可共用 GPU，每段视频保持顺序跟踪。终端和 `progress.json` 显示进度。
中断后加 `--resume` 接续；`--segments G3-6` 可单独运行一个示例。

## 使用正式数据

将 Zenodo 上传包 `PEMFlow_data.zip` 解压，把其中 `PEM-Flow/` 合并到代码库同名目录。
两版正式轨迹分别位于 `data/automatic/trajectory/`（自动处理）和
`data/revised/trajectory/`（人工核验后）。相邻目录保存运动学、行来源和轨迹图。

计算核验后数据的下游指标和质量指标：

```powershell
.venv\Scripts\python -m pemflow metrics --input ../data/revised --output ../outputs/downstream --workers 2
.venv\Scripts\python -m pemflow evaluate --input ../data/revised --automatic ../data/automatic --output ../outputs/evaluation
.venv\Scripts\python examples/score_archived_ap.py --output ../outputs/ap
```

AP 示例读取包内的预测记录与人工标注。使用随数据提供的图像观测和稳像矩阵计算轨迹，
并执行人工修订记录：

```powershell
.venv\Scripts\python examples/reproject_cached.py --output ../outputs/automatic
.venv\Scripts\python -m pemflow revise --input ../outputs/automatic --output ../outputs/revised
```

`corrections/final_rules.json` 保存 77 条人工判断，使用自动数据中的源 ID 和包含首尾的帧范围。
删除操作过滤指定行；ID 映射同时赋值；人工提交的框使用对应矩阵投影。
受影响的完整轨迹采用同一运动学方法计算，逐行记录保存在 `data/revision_records/`。

## 配置与输出

`configs/demo_dataset.json` 描述三个示例；`configs/dataset.json` 保存 19 段正式数据的配置。
每项包含视频文件名、帧数、相机参数、三维/像素控制点和标定参考帧。
`configs/pipeline.json` 保存 ROI 与目标形状参数。
类别参考平面高度为行人 0.72 m、电动自行车 0.20 m、机动车 0.60 m。

处理其他视频时，提供对应视频的配置 JSON，填写其相机和控制点、图像尺寸、帧数、
文件大小和 SHA-256：

```powershell
.venv\Scripts\python -m pemflow run --dataset configs/my_dataset.json --video-root C:/my_videos --output ../outputs/my_run
```

该 JSON 与 `inference.json`、`bytetrack_pemflow.yaml` 放在同一配置目录，标定参考帧索引应位于视频内。
程序按所选配置处理，输出 `trajectory/`、`motion/`、`lineage/` 和运行记录。
显式相对路径以当前工作目录为基准。

轨迹表字段：

```text
frame,id,class,real_x,real_y,velocity,dist
```

位置和距离单位为米，速度为 m/s；类别 0 为行人、9 为电动自行车、3 为机动车。
位置、速度和加速度使用二阶 Savitzky–Golay 拟合，最大窗口 31 帧；25–30 帧连续段
采用合法的最大奇数窗口，更短的段保留位置和缺失导数。G07 两段的文件名为
`G3-7.csv`、`G3-7_hmi.csv`，帧号与身份各自独立。

`src/pemflow/pipeline/` 负责视频处理，`revision/` 执行人工记录，`downstream/` 计算运动和 ROI 指标，
`evaluation/` 评价人工标注。`cli.py` 提供命令入口，`execution.py` 调度任务，`io.py` 处理路径。
详细定义见[方法说明](docs/methods.md)、[字段说明](docs/data_dictionary.md)和[许可说明](THIRD_PARTY_NOTICES.md)。
