import os
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from model_definitions.tracknet2 import TrackNetV2

def preprocess_frame(frame, transform):
    frame = transform(frame)
    return frame

def predict_ball_position(prev_positions, width, height):
    if len(prev_positions) < 3:
        return None
    p_t = prev_positions[-1]
    a_t = p_t - 2 * prev_positions[-2] + prev_positions[-3]
    v_t = p_t - prev_positions[-2] + a_t
    predicted_position = p_t + v_t + 0.5 * a_t
    predicted_position = np.clip(predicted_position, [0, 0], [width, height])
    return predicted_position


def run_inference(weights, input_path, overlay=False):
    # Step 1: Configuration and Initialization
    # =======================================================
    print("Performing model configuration and initialization...")
    config = {
        "name": "tracknetv2",
        "frames_in": 3,
        "frames_out": 3,
        "inp_height": 288,
        "inp_width": 512,
        "out_height": 288,
        "out_width": 512,
        "bilinear": True,
        "halve_channel": False,
        "mode": "nearest",
        "rgb_diff": False,
        "out_scales": [0],
        "model_path": f"model_weights/tracknetv2_{weights}_best.pth.tar",
    }
    # 自动选择设备（GPU 或 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device selected: {device}")

    # 定义图像预处理转换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['inp_height'], config['inp_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Step 2: Load the model and weights
    # =======================================================
    print(f"Loading model: {config['model_path']}")
    model = TrackNetV2(n_channels=9, n_classes=3, bilinear=config['bilinear'], mode=config['mode'], halve_channel=config['halve_channel']).to(device)
    checkpoint = torch.load(config['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    # Step 3: Video Input/Output Setup
    # =======================================================
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_path}")
        return

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_video_path = os.path.join(os.path.dirname(input_path), f"{base_name}_output_tracknetv2.mp4")
    output_csv_path = os.path.join(os.path.dirname(input_path), f"{base_name}_output_tracknetv2.csv")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    print(f"Output video path: {output_video_path}")
    print(f"Output CSV path: {output_csv_path}")
    print(f"Total frames to process: {total_frames}")

    coordinates = []
    frame_number = 0
    frames_buffer = []
    prev_positions = []
    
    # 初始化计时变量
    start_time = time.time()
    processed_frames = 0
    
    # === 新增代码: 初始化拖尾效果的参数 ===
    trace_points = []
    trace_length = 20 # 拖尾的长度，即保留多少个历史点
    # ==================================

    # Step 4: Video Frame Processing Loop with tqdm
    # =======================================================
    print("Starting video frame processing...")
    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames_buffer.append(frame)
            if len(frames_buffer) == config['frames_in']:
                
                frames_processed = [preprocess_frame(f, transform) for f in frames_buffer]
                input_tensor = torch.cat(frames_processed, dim=0).unsqueeze(0).to(device)

                # Step 5: Model Inference and Post-processing
                # =======================================================
                with torch.no_grad():
                    outputs = model(input_tensor)[0]
                
                for i in range(config['frames_out']):
                    output = outputs[0][i]
                    output = torch.sigmoid(output)
                    heatmap = output.squeeze().cpu().numpy()
                    heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
                    heatmap = (heatmap > 0.5).astype(np.float32) * heatmap

                    current_frame = frames_buffer[i].copy() # 使用 .copy() 以确保在绘制时不会修改原始帧
                    
                    if overlay:
                        heatmap_vis = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
                        current_frame = cv2.addWeighted(current_frame, 0.6, heatmap_colored, 0.4, 0)
                        
                    # Step 6: Find and Track Ball Position
                    # =======================================================
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

                    # Step 7: Draw circle and write to video
                    # =======================================================
                    # === 新增代码: 更新和绘制拖尾效果 ===
                    if detected:
                        trace_points.append((int(center_x), int(center_y)))
                        # 控制拖尾长度
                        if len(trace_points) > trace_length:
                            trace_points.pop(0)
                        
                        # 绘制拖尾
                        for point in trace_points:
                            cv2.circle(current_frame, point, 3, (0, 255, 255), -1) # 绘制黄色实心圆
                        
                        # 绘制当前的球
                        cv2.circle(current_frame, (int(center_x), int(center_y)), 10, (0, 255, 0), 2)
                        coordinates.append([frame_number, 1, center_x, center_y, confidence])
                    else:
                        # 如果没有检测到球，则不更新拖尾
                        coordinates.append([frame_number, 0, 0, 0, 0])

                    out.write(current_frame)
                    
                    frame_number += 1
                frames_buffer = []

                processed_frames += config['frames_out']
                pbar.update(config['frames_out'])
                
                if processed_frames % 30 == 0 and time.time() - start_time > 0:
                    current_fps = processed_frames / (time.time() - start_time)
                    pbar.set_postfix_str(f"FPS: {current_fps:.2f}")
    
    # Step 8: Release resources and save CSV
    # =======================================================
    end_time = time.time()
    total_time = end_time - start_time
    
    if total_time > 0:
        avg_fps = total_frames / total_time
        print(f"\nTotal frames processed: {total_frames}")
        print(f"Total time elapsed: {total_time:.2f} seconds")
        print(f"Average FPS (including all processing): {avg_fps:.2f}")
    else:
        print("\nProcessing time was too short to calculate FPS.")

    print("All resources released.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Saving coordinates to CSV file...")
    coordinates_df = pd.DataFrame(coordinates, columns=["frame_number", "detected", "x", "y", "confidence (blob sum)"])
    coordinates_df.to_csv(output_csv_path, index=False)
    print("CSV file saved.")