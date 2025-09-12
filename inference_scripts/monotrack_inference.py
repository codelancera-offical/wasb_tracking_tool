import os
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import time
from model_definitions.monotrack import MonoTrack

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

def run_inference(weights, input_path, overlay=False):
    # Step 1: Configuration and Initialization
    # =======================================================
    print("Performing model configuration and initialization for MonoTrack...")
    config = {
        "name": "monotrack",
        "frames_in": 3,
        "frames_out": 3,
        "inp_height": 288,
        "inp_width": 512,
        "out_height": 288,
        "out_width": 512,
        "out_scales": [0],
        "rgb_diff": False,
        "bilinear": False,
        "halve_channel": True,
        "mode": 'nearest',
        "model_path": f"model_weights/monotrack_{weights}_best.pth.tar",
    }
    
    # Auto-select device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device selected: {device}")

    # Define the transformation to preprocess the frames
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['inp_height'], config['inp_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Step 2: Load the model and weights
    # =======================================================
    print(f"Loading model: {config['model_path']}")
    model = MonoTrack(n_channels=9, n_classes=3, 
                      bilinear=config['bilinear'], 
                      halve_channel=config['halve_channel']).to(device)
    try:
        checkpoint = torch.load(config['model_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except FileNotFoundError:
        print(f"Error: Model file not found at {config['model_path']}. Please check the file path.")
        return
    model.eval()  # Set model to evaluation mode

    # Step 3: Video Input/Output Setup
    # =======================================================
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_path}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_video_path = os.path.join(os.path.dirname(input_path), f"{base_name}_output_monotrack.mp4")
    output_csv_path = os.path.join(os.path.dirname(input_path), f"{base_name}_output_monotrack.csv")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Output video path: {output_video_path}")
    print(f"Output CSV path: {output_csv_path}")
    print(f"Total frames to process: {total_frames}")

    coordinates = []
    frame_number = 0
    frames_buffer = []
    
    # === 新增代码: 初始化拖尾效果和计时变量 ===
    trace_points = []
    trace_length = 20
    prev_positions = [] # 用于predict_ball_position
    start_time = time.time()
    # ====================================================

    # Step 4: Video Frame Processing Loop with detailed logging
    # =======================================================
    print("Starting video frame processing loop...")
    while True:
        # 4a. Read frame
        ret, frame = cap.read()
        if not ret:
            print("Finished reading all video frames.")
            break
        
        # Log progress every 100 frames
        if (frame_number + 1) % 100 == 0:
            elapsed_time = time.time() - start_time
            current_fps = (frame_number + 1) / elapsed_time if elapsed_time > 0 else 0
            print(f"Processing frame {frame_number + 1}/{total_frames} | Current FPS: {current_fps:.2f}")

        frames_buffer.append(frame)

        # 确保帧缓冲区有3帧再进行推理
        if len(frames_buffer) < config['frames_in']:
            # 对于前两帧，不进行推理，直接写入原始帧并记录
            current_frame = frame.copy()
            out.write(current_frame)
            coordinates.append([frame_number, 0, 0, 0])
            frame_number += 1
            continue
        
        if len(frames_buffer) > config['frames_in']:
             # 移除最旧的一帧，确保缓冲区大小为3
            frames_buffer.pop(0)

        # Step 5: Model Inference and Post-processing
        # =======================================================
        # 5a. Preprocess frames
        frames_processed = [preprocess_frame(f, transform) for f in frames_buffer]
        input_tensor = torch.cat([frames_processed[0], frames_processed[1], frames_processed[2]], dim=0).unsqueeze(0).to(device)
        
        # 5b. Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)[0]
        
        # MonoTrack模型只输出1个热图，对应于缓冲区中的最后一帧
        output = outputs[0][2]
        
        # 5c. Post-process the output
        output = torch.sigmoid(output)
        heatmap = output.squeeze().cpu().numpy()
        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
        heatmap = (heatmap > 0.5).astype(np.float32) * heatmap
        
        current_frame = frames_buffer[2].copy()
        
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

        # Step 7: Draw circle, trail, and write to video
        # =======================================================
        if detected:
            trace_points.append((int(center_x), int(center_y)))
            if len(trace_points) > trace_length:
                trace_points.pop(0)
            
            for point in trace_points:
                cv2.circle(current_frame, point, 3, (0, 255, 255), -1)
            
            cv2.circle(current_frame, (int(center_x), int(center_y)), 10, (0, 255, 0), 2)
            coordinates.append([frame_number, 1, center_x, center_y, confidence])
        else:
            coordinates.append([frame_number, 0, 0, 0])
        
        out.write(current_frame)
        frame_number += 1
        
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
    # NOTE: MonoTrack's output doesn't have a 'confidence' score in this version,
    # so we'll use a placeholder for consistency with other scripts.
    coordinates_df = pd.DataFrame(coordinates, columns=["frame_number", "detected", "x", "y"])
    coordinates_df.to_csv(output_csv_path, index=False)
    print("CSV file saved.")