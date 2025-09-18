import argparse
import os
import sys
from pathlib import Path

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
    # 1. 修改 --input 的帮助信息，使其更清晰
    parser.add_argument("--input", type=str, required=True, help="Specify the input video file or a folder containing video files.")
    # 2. 新增 --output 参数，用于指定输出的总目录，并设置默认值为 'outputs'
    parser.add_argument("--output", type=str, default="outputs", help="Specify the base directory for all output results. Defaults to 'outputs'.")
    parser.add_argument('--overlay', action='store_true', help='Overlay heatmap on the original frame')
    
    args = parser.parse_args()

    # --- 3. 核心逻辑修改：处理输入路径，使其能接受文件或文件夹 ---
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: The input path '{args.input}' does not exist.")
        sys.exit(1)

    # 创建一个列表来存放所有待处理的视频文件
    video_files_to_process = []
    supported_extensions = ['.mp4', '.avi', '.mov', '.mkv'] # 支持的视频格式

    if input_path.is_file():
        # 如果输入是单个文件
        if input_path.suffix.lower() in supported_extensions:
            video_files_to_process.append(input_path)
        else:
            print(f"Error: Input file '{input_path}' is not a supported video format {supported_extensions}.")
            sys.exit(1)
            
    elif input_path.is_dir():
        # 如果输入是一个文件夹，则遍历该文件夹
        print(f"Input is a directory. Searching for videos in: {input_path}")
        for file_path in input_path.iterdir():
            if file_path.suffix.lower() in supported_extensions:
                video_files_to_process.append(file_path)
    
    if not video_files_to_process:
        print(f"Error: No supported video files found in '{args.input}'.")
        sys.exit(1)
        
    print(f"Found {len(video_files_to_process)} video(s) to process.")
    # -----------------------------------------------------------------

    # 选择推理函数（只需选择一次）
    inference_function = MODEL_INFERENCE_MAP[args.model]

    # --- 4. 遍历所有找到的视频文件并进行处理 ---
    for video_path in video_files_to_process:
        print("-" * 50)
        print(f"Processing video: {video_path.name}")

        # 提取文件名，不含扩展名 (e.g., "my_video")
        video_stem = video_path.stem
        
        # 在用户指定的输出目录下，为当前视频创建一个专属的结果目录
        # 使用 Path 对象来构建路径
        base_output_dir = Path(args.output)
        output_dir = base_output_dir / f"{video_stem}_{args.model}_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 构建标准化的输出文件完整路径
        output_video_path = output_dir / f"{video_stem}_tracked.mp4"
        output_csv_path = output_dir / f"{video_stem}_coordinates.csv"
        
        print(f"Results will be saved to: {output_dir}")

        # 调用推理函数，传入当前视频的路径和对应的输出路径
        try:
            inference_function(
                weights=args.weights, 
                input_path=str(video_path), # 传入当前处理的视频路径
                overlay=args.overlay,
                output_video_path=str(output_video_path),
                output_csv_path=str(output_csv_path)
            )
            print(f"Successfully processed {video_path.name}")
        except Exception as e:
            print(f"An error occurred while processing {video_path.name}: {e}")
    
    print("-" * 50)
    print("All videos have been processed.")


if __name__ == "__main__":
    main()
