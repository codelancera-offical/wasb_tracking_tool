import os
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import time
from model_definitions.wasb import HRNet
# ===================================================================================
# 修改 1: 引入deque
# 目的: 使用一个高效的双端队列来存储轨迹点。设置maxlen后，它能自动维护一个
#       固定长度的轨迹历史，当新点加入时，最旧的点会自动被移除。
from collections import deque
# ===================================================================================

def preprocess_frame(frame, transform):
    return transform(frame)

def predict_ball_position(prev_positions, width, height):
    if len(prev_positions) < 3:
        return None
    p_t = prev_positions[-1]
    a_t = p_t - 2 * prev_positions[-2] + prev_positions[-3]
    v_t = p_t - prev_positions[-2] + a_t
    predicted_position = p_t + v_t + 0.5 * a_t
    predicted_position = np.clip(predicted_position, [0, 0], [width, height])
    return predicted_position

# ===================================================================================
# 修改 2: 更新函数签名
# 目的: 让函数能够接收由 main.py 生成和传入的、标准化的输出文件路径，
#       从而将文件I/O的控制权上交给主流程脚本。
def run_inference(weights, input_path, output_video_path, output_csv_path, overlay=False):
# ===================================================================================

    # Step 1: Configuration and Initialization (这部分未做修改)
    print("Performing model configuration and initialization for WASB (HRNet)...")
    config = {
        "name": "hrnet", "frames_in": 3, "frames_out": 3, "inp_height": 288, "inp_width": 512,
        "out_height": 288, "out_width": 512, "rgb_diff": False, "out_scales": [0],
        "MODEL": {
            "EXTRA": {
                "FINAL_CONV_KERNEL": 1, "PRETRAINED_LAYERS": ['*'],
                "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
                "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": 'BOTTLENECK', "NUM_BLOCKS": [1], "NUM_CHANNELS": [32], "FUSE_METHOD": 'SUM'},
                "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": 'BASIC', "NUM_BLOCKS": [2, 2], "NUM_CHANNELS": [16, 32], "FUSE_METHOD": 'SUM'},
                "STAGE3": {"NUM_MODULES": 1, "NUM_BRANCHES": 3, "BLOCK": 'BASIC', "NUM_BLOCKS": [2, 2, 2], "NUM_CHANNELS": [16, 32, 64], "FUSE_METHOD": 'SUM'},
                "STAGE4": {"NUM_MODULES": 1, "NUM_BRANCHES": 4, "BLOCK": 'BASIC', "NUM_BLOCKS": [2, 2, 2, 2], "NUM_CHANNELS": [16, 32, 64, 128], "FUSE_METHOD": 'SUM'},
                "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_BASIC_BLOCKS": 2}
            }, "INIT_WEIGHTS": True
        },
        "model_path": f"model_weights/wasb_{weights}_best.pth.tar",
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device selected: {device}")
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((config['inp_height'], config['inp_width'])), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Step 2: Load the model and weights (这部分未做修改)
    print(f"Loading model: {config['model_path']}")
    model = HRNet(cfg=config).to(device)
    try:
        checkpoint = torch.load(config['model_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except FileNotFoundError:
        print(f"Error: Model file not found at {config['model_path']}. Please check the file path.")
        return
    model.eval()

    # Step 3: Video Input/Output Setup
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ===================================================================================
    # 修改 3: 移除内部路径生成逻辑
    # 目的: 删除原先在脚本内部根据输入路径硬编码生成输出路径的代码，
    #       使其完全依赖外部传入的参数。
    # base_name = os.path.splitext(os.path.basename(input_path))[0]
    # output_video_path = os.path.join(os.path.dirname(input_path), f"{base_name}_output_wasb.mp4")
    # output_csv_path = os.path.join(os.path.dirname(input_path), f"{base_name}_output_wasb.csv")
    # ===================================================================================
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Output video will be saved to: {output_video_path}")
    print(f"Output CSV will be saved to: {output_csv_path}")
    print(f"Total frames to process: {total_frames}")

    coordinates = []
    frame_number = 0
    frames_buffer = []
    prev_positions = []
    
    # ===================================================================================
    # 修改 4: 初始化轨迹队列
    # 目的: 创建一个deque对象来存储轨迹点，而不是普通的list。
    #       这里的 trace_length 可以灵活调整，数值越大，轨迹拖尾越长。
    trace_length = 15  # 轨迹点的最大数量
    trace_points = deque(maxlen=trace_length) 
    # ===================================================================================
    start_time = time.time()
    
    print("Starting video frame processing loop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished reading all video frames.")
            break
        
        if (frame_number + 1) % 100 == 0:
            elapsed_time = time.time() - start_time
            current_fps = (frame_number + 1) / elapsed_time if elapsed_time > 0 else 0
            print(f"Processing frame {frame_number + 1}/{total_frames} | Current FPS: {current_fps:.2f}")

        frames_buffer.append(frame)
        if len(frames_buffer) == config['frames_in']:
            frames_processed = [preprocess_frame(f, transform) for f in frames_buffer]
            input_tensor = torch.cat(frames_processed, dim=0).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)[0]

            for i in range(config['frames_out']):
                output = outputs[0][i]
                output = torch.sigmoid(output)
                heatmap = output.squeeze().cpu().numpy()
                heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
                heatmap = (heatmap > 0.5).astype(np.float32) * heatmap

                current_frame = frames_buffer[i].copy()
                if overlay:
                    heatmap_vis = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
                    current_frame = cv2.addWeighted(current_frame, 0.6, heatmap_colored, 0.4, 0)
                
                # Ball detection logic (未做修改)
                num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats((heatmap > 0).astype(np.uint8), connectivity=8)
                blob_centers = []
                for j in range(1, num_labels):
                    mask = labels_im == j
                    blob_sum = heatmap[mask].sum()
                    if blob_sum > 0:
                        center_x = np.sum(np.where(mask)[1] * heatmap[mask]) / blob_sum
                        center_y = np.sum(np.where(mask)[0] * heatmap[mask]) / blob_sum
                        blob_centers.append((center_x, center_y, blob_sum))
                
                detected = False
                center_x, center_y, confidence = 0, 0, 0
                if blob_centers:
                    predicted_position = predict_ball_position(prev_positions, width, height)
                    if predicted_position is not None:
                        distances = [np.sqrt((x - predicted_position[0]) ** 2 + (y - predicted_position[1]) ** 2) for x, y, _ in blob_centers]
                        closest_blob_idx = np.argmin(distances)
                        center_x, center_y, confidence = blob_centers[closest_blob_idx]
                    else:
                        blob_centers.sort(key=lambda x: x[2], reverse=True)
                        center_x, center_y, confidence = blob_centers[0]
                    detected = True
                    prev_positions.append(np.array([center_x, center_y]))
                    if len(prev_positions) > 3:
                        prev_positions.pop(0)

                # ===================================================================================
                # 修改 5: 更新为 "点状" 轨迹可视化逻辑
                # 目的: 根据最新需求，不再连接轨迹点，而是将它们作为独立的点绘制出来，
                #       并且即使当前帧未检测到球，历史轨迹点依然会显示。

                # 步骤 5a: 绘制历史轨迹点 (无论是否检测到都执行)
                # 遍历deque中的每一个历史坐标点
                for point in trace_points:
                    # 为每个历史点画一个较小的黄色实心圆圈
                    cv2.circle(current_frame, point, 3, (0, 255, 255), -1)

                # 步骤 5b: 如果当前帧检测到了球，则更新轨迹并绘制当前点
                if detected:
                    new_point = (int(center_x), int(center_y))
                    # 将新坐标点加入轨迹队列
                    trace_points.append(new_point)
                    
                    # 在当前位置画一个醒目的绿色大圆圈，以突出显示
                    cv2.circle(current_frame, new_point, 10, (0, 255, 0), 2)
                    
                    # 记录数据
                    coordinates.append([frame_number, 1, center_x, center_y, confidence])
                else:
                    # 如果没检测到，就不更新轨迹点，只记录数据
                    coordinates.append([frame_number, 0, 0, 0, 0])
                # ===================================================================================
                
                out.write(current_frame)
                frame_number += 1
            
            frames_buffer = []

    # Step 8: Release resources and save CSV (这部分未做修改)
    end_time = time.time()
    total_time = end_time - start_time
    if total_time > 0:
        avg_fps = (frame_number + 1) / total_time
        print(f"\nTotal frames processed: {frame_number + 1}")
        print(f"Total time elapsed: {total_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
    else:
        print("\nProcessing time was too short to calculate FPS.")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Saving coordinates to CSV file...")
    coordinates_df = pd.DataFrame(coordinates, columns=["frame_number", "detected", "x", "y", "confidence"])
    coordinates_df.to_csv(output_csv_path, index=False)
    print("CSV file saved successfully.")