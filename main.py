# main.py

import argparse
import os
import sys
from pathlib import Path # 1. 引入Pathlib，用于更优雅地处理文件路径

# Import inference functions from other scripts
from inference_scripts.deepball_inference import run_inference as deepball_inference
from inference_scripts.deepball_large_inference import run_inference as deepball_large_inference
from inference_scripts.ballseg_inference import run_inference as ballseg_inference
from inference_scripts.monotrack_inference import run_inference as monotrack_inference
from inference_scripts.restracknetv2_inference import run_inference as restracknetv2_inference
from inference_scripts.tracknetv2_inference import run_inference as tracknetv2_inference
from inference_scripts.wasb_inference import run_inference as wasb_inference

# Map models to their corresponding inference functions
MODEL_INFERENCE_MAP = {
    "deepball": deepball_inference,
    "deepball-large": deepball_large_inference,
    "ballseg": ballseg_inference,
    "monotrack": monotrack_inference,
    "restracknetv2": restracknetv2_inference,
    "tracknetv2": tracknetv2_inference,
    "wasb": wasb_inference,
}

def main():
    parser = argparse.ArgumentParser(description="Run inference on different models with specified weights and input type.")
    parser.add_argument("--weights", type=str, choices=["tennis", "badminton", "soccer"], required=True, help="Specify the weights to use: 'tennis' or 'badminton' or 'soccer'.")
    parser.add_argument("--model", type=str, choices=list(MODEL_INFERENCE_MAP.keys()), required=True, help="Specify the model to use.")
    parser.add_argument('--overlay', action='store_true', help='Overlay heatmap on the original frame')    
    parser.add_argument("--input", type=str, required=True, help="Specify the input file or folder.")
    
    args = parser.parse_args()

    # Check if input path exists
    input_path = Path(args.input) # 2. 将输入路径字符串转换为Path对象
    if not input_path.exists():
        print(f"Error: The input path '{args.input}' does not exist.")
        sys.exit(1)

    # --- 3. 新增的核心逻辑：自动创建和管理输出路径 ---
    video_stem = input_path.stem  # 提取文件名，不含扩展名 (e.g., "my_video")
    
    # 在 'outputs' 文件夹下创建一个专属的结果目录
    output_dir = Path(f"outputs/{video_stem}_{args.model}_results")
    output_dir.mkdir(parents=True, exist_ok=True) # parents=True确保能创建多级目录, exist_ok=True避免目录已存在时报错

    # 构建标准化的输出文件完整路径
    output_video_path = output_dir / f"{video_stem}_tracked.mp4"
    output_csv_path = output_dir / f"{video_stem}_coordinates.csv"
    
    print(f"Input video: {input_path}")
    print(f"Results will be saved to: {output_dir}")
    # ----------------------------------------------------

    # Select the appropriate inference function
    inference_function = MODEL_INFERENCE_MAP[args.model]

    # 4. 调用推理函数，并传入新的输出路径参数
    # 注意：这里我们假设所有推理函数都将遵循这个新的接口
    inference_function(
        weights=args.weights, 
        input_path=str(input_path), # 传入字符串形式的路径
        overlay=args.overlay,
        output_video_path=str(output_video_path),
        output_csv_path=str(output_csv_path)
    )

if __name__ == "__main__":
    main()