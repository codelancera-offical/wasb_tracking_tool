# WASB Tracking Tool

该工具的核心功能是接收一个网球视频，并输出两份产物：

* 一份经过处理的视频文件，其中网球的轨迹被清晰地可视化出来。
* 一份 `.csv` 数据文件，记录了每一帧中网球的精确坐标和检测置信度。

视频推理的效果如下所示：
![](./final_output.gif)

## 使用方法

### 1. 克隆代码仓库

```sh
git clone https://github.com/codelancera-offical/wasb_tracking_tool.git
```

### 2. 环境配置

1. 用conda新建一个python==3.12.11的环境
2. 终端进入文件夹，执行`pip install -r requirements.txt`

### 3. 开始推理

在`main.py`所在目录下，按照如下命令格式在终端执行：

```sh
python main.py --model wasb --weights tennis --input <输入路径> --output <输出路径>
```
- `-input`: **必需参数**。指定输入的路径。
    - 可以是**单个视频文件**的完整路径 (例如: `./videos/my_match.mp4`)。
    - 也可以是包含多个视频文件 (如 `.mp4`, `.avi`) 的**文件夹路径** (例如: `./videos/`)。
- `-output`: **可选参数**。指定存放所有结果的总目录。
    - 如果**不提供**此参数，结果将默认保存在项目根目录下的 `outputs/` 文件夹中。
    - 如果提供，则所有结果都会存放在您指定的路径下，其中每个视频的结果会包含一个csv文件和一个推理后的可视化视频。

### TODO

- [ ] 遮挡轨迹还原
- [ ] 排除相似物对球体轨迹的干扰
- [ ] 关键事件(击球，弹跳，下网)检测
