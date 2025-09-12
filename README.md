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
python main.py --weights tennis --model wasb --input <视频文件路径\>
```

结果以文件夹形式会存放到outputs/中，其中会包含一个csv文件和一个推理后的可视化视频。

### TODO

- [ ] 遮挡轨迹还原
- [ ] 排除相似物对球体轨迹的干扰
- [ ] 关键事件(击球，弹跳，下网)检测