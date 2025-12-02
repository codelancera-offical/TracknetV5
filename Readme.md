# TrackNetV5 批量推理脚本使用说明

该脚本用于对文件夹内的所有视频（.mp4/.mov）进行批量推理，并生成轨迹视频、对比视频及坐标数据 CSV。

## 1\. 安装运行环境

```
pip install -r requirements.txt
```

## 2\. 基础命令

```bash
python inference.py <视频文件夹路径> <权重文件路径> --arch <模型版本>
```

## 3\. 参数说明

| 参数 | 类型 | 是否必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `input_dir` | 位置参数 | **是** | 存放待测试视频的**文件夹**路径 |
| `weights_path` | 位置参数 | **是** | `.pth` 模型权重文件的路径 |
| `--arch` | 选项参数 | **是** | 模型架构版本，可选：`v2`, `v4`, `v5` |
| `--device` | 选项参数 | 否 | 运行设备，默认 `cuda:0`，也可选 `cpu` |
| `--threshold` | 选项参数 | 否 | 置信度阈值 (0.0 - 1.0)，默认 `0.5` |

## 4\. 使用示例

### 示例 1：使用 TrackNetV5 进行推理（推荐）

```bash
python inference.py ./data/test_videos ./weights/tracknet_v5_best.pth --arch v5
```

### 示例 2：调整阈值 (处理漏检/误检)

如果觉得球没检测出来，可以降低阈值到 0.3：

```bash
python inference.py ./data/test_videos ./weights/tracknet_v5_best.pth --arch v5 --threshold 0.3
```

## 5\. 输出结果

脚本运行后，会在 `input_dir` 下自动生成一个以模型版本命名的文件夹（例如 `v5/`），结构如下：

  * **`_summary_report_v5.csv`**: 所有视频的汇总统计表（包含检测率等）。
  * **`视频文件名/`**: 每个视频的独立结果文件夹，包含：
      * `_trajectory.mp4`: 纯轨迹视频。
      * `_comparison.mp4`: 轨迹与热力图对比视频。
      * `_data.csv`: 逐帧坐标数据。
