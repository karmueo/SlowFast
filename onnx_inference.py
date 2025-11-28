import argparse
import os
import cv2
import numpy as np
import onnxruntime as ort
from slowfast.config.defaults import get_cfg

def parse_args():
    parser = argparse.ArgumentParser(description="Run ONNX inference on a video frame folder")
    parser.add_argument("--onnx_model", required=True, help="Path to ONNX model")
    parser.add_argument("--input_folder", required=True, help="Path to video frames folder")
    parser.add_argument("--cfg", required=True, help="Path to config file")
    return parser.parse_args()

def load_config(cfg_file):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    return cfg

def preprocess(cfg, folder_path):
    # Parameters from config
    num_frames = cfg.DATA.NUM_FRAMES
    sampling_rate = cfg.DATA.SAMPLING_RATE
    crop_size = cfg.DATA.TEST_CROP_SIZE
    mean = np.array(cfg.DATA.MEAN, dtype=np.float32)
    std = np.array(cfg.DATA.STD, dtype=np.float32)

    # Get image files
    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                          if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
    
    if not image_files:
        raise ValueError(f"No images found in {folder_path}")

    total_frames = len(image_files)
    clip_length = num_frames * sampling_rate
    
    # Temporal sampling (Center crop in time)
    start_frame = 0
    if total_frames > clip_length:
        start_frame = (total_frames - clip_length) // 2
    
    frames = []
    for i in range(num_frames):
        frame_idx = start_frame + i * sampling_rate
        # Loop padding if not enough frames
        if frame_idx >= total_frames:
            frame_idx = frame_idx % total_frames
            
        img_path = image_files[frame_idx]
        img = cv2.imread(img_path)
        if img is None:
             # Handle missing frames by duplicating last or zeros if empty
             if frames:
                 frames.append(frames[-1])
             else:
                 frames.append(np.zeros((crop_size, crop_size, 3), dtype=np.uint8))
             continue
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Spatial preprocessing (Resize short side + Center Crop)
        h, w, _ = img.shape
        if h < w:
            new_h = crop_size
            new_w = int(w * crop_size / h)
        else:
            new_w = crop_size
            new_h = int(h * crop_size / w)
        
        img = cv2.resize(img, (new_w, new_h))
        
        # Center crop
        start_x = (new_w - crop_size) // 2
        start_y = (new_h - crop_size) // 2
        img = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
        
        frames.append(img)

    # Stack frames: T x H x W x C
    tensor = np.stack(frames)
    
    # Normalize with Mean/Std
    tensor = tensor.astype(np.float32) / 255.0
    tensor = (tensor - mean) / std
    
    # Transpose to C x T x H x W
    tensor = tensor.transpose(3, 0, 1, 2)
    
    # Add batch dimension: 1 x C x T x H x W
    tensor = np.expand_dims(tensor, axis=0)
    
    return tensor.astype(np.float32)

def main():
    args = parse_args()
    
    # Load config
    cfg = load_config(args.cfg)
    
    # Preprocess
    print(f"Preprocessing images from {args.input_folder}...")
    input_tensor = preprocess(cfg, args.input_folder)
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Load ONNX model
    print(f"Loading ONNX model from {args.onnx_model}...")
    ort_session = ort.InferenceSession(args.onnx_model)
    
    # Get input name
    input_name = ort_session.get_inputs()[0].name
    
    # Run inference
    print("Running inference...")
    outputs = ort_session.run(None, {input_name: input_tensor})
    
    # Output results
    output = outputs[0]
    print("Inference Results:")
    print(output)
    
    if output.shape[1] == 2:
        print(f"Class 0: {output[0][0]}")
        print(f"Class 1: {output[0][1]}")

if __name__ == "__main__":
    main()
