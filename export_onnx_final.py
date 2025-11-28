#!/usr/bin/env python3
"""
Export X3D model to ONNX with real data testing support.
Uses legacy ONNX exporter for maximum accuracy.
"""

import argparse
import numpy as np
import torch
import onnxruntime as ort
import cv2
from pathlib import Path

try:
    import onnxsim
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False

from slowfast.config.defaults import get_cfg
from slowfast.models import build_model
from slowfast.utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export X3D model to ONNX and test with real data"
    )
    parser.add_argument(
        "--cfg",
        default="configs/Custom/X3D_M_custom.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        default="output/custom_x3d_3/checkpoints/checkpoint_epoch_00150.pyth",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--output",
        default="output/custom_x3d_3/model.onnx",
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--test_video",
        default=None,
        help="Path to video file for testing"
    )
    parser.add_argument(
        "--test_frames",
        default=None,
        help="Path to directory containing video frames for testing"
    )
    parser.add_argument(
        "--class_names",
        default=None,
        help="Comma-separated class names (e.g., 'class0,class1')"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX model using onnx-simplifier"
    )
    parser.add_argument(
        "--use_dynamo",
        action="store_true",
        help="Use new dynamo-based exporter (may have accuracy issues)"
    )
    parser.add_argument(
        "--export_softmax",
        action="store_true",
        help="Export additional model with softmax layer for inference"
    )
    return parser.parse_args()


def simplify_onnx_model(input_path, output_path):
    """Simplify ONNX model if onnx-simplifier is available."""
    if not ONNXSIM_AVAILABLE:
        print("⚠ onnx-simplifier not installed, skipping simplification")
        print("  Install: pip install onnx-simplifier")
        return False
    
    try:
        import onnx
        model = onnx.load(input_path)
        model_simp, check = onnxsim.simplify(model)
        if check:
            onnx.save(model_simp, output_path)
            print(f"✓ Simplified model saved to: {output_path}")
            return True
        else:
            print("✗ Simplification failed validation")
            return False
    except Exception as e:
        print(f"✗ Simplification error: {e}")
        return False


def load_video_frames(video_path, num_frames, sampling_rate, target_size):
    """Load frames from video file with SlowFast standard sampling."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Spatial preprocessing (Resize short side + Center Crop)
            h, w, _ = frame.shape
            if h < w:
                new_h = target_size
                new_w = int(w * target_size / h)
            else:
                new_w = target_size
                new_h = int(h * target_size / w)
            
            frame = cv2.resize(frame, (new_w, new_h))
            
            # Center crop
            start_x = (new_w - target_size) // 2
            start_y = (new_h - target_size) // 2
            frame = frame[start_y:start_y+target_size, start_x:start_x+target_size]
            
            frames.append(frame)
        else:
            # Handle read failure
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((target_size, target_size, 3), dtype=np.uint8))
    
    cap.release()
    return np.array(frames)


def load_frames_from_directory(frames_dir, num_frames, sampling_rate, target_size):
    """Load frames from directory with SlowFast standard sampling."""
    frame_dir = Path(frames_dir)
    image_files = sorted([
        f for f in frame_dir.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
    ])
    
    if not image_files:
        raise ValueError(f"No image files found in: {frames_dir}")
    
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
            
        frame = cv2.imread(str(image_files[frame_idx]))
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Spatial preprocessing (Resize short side + Center Crop)
            h, w, _ = frame.shape
            if h < w:
                new_h = target_size
                new_w = int(w * target_size / h)
            else:
                new_w = target_size
                new_h = int(h * target_size / w)
            
            frame = cv2.resize(frame, (new_w, new_h))
            
            # Center crop
            start_x = (new_w - target_size) // 2
            start_y = (new_h - target_size) // 2
            frame = frame[start_y:start_y+target_size, start_x:start_x+target_size]
            
            frames.append(frame)
        else:
             # Handle read failure
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((target_size, target_size, 3), dtype=np.uint8))
    
    return np.array(frames)


def preprocess_frames(frames, cfg):
    """
    Preprocess frames for model input with Mean/Std normalization.
    frames: (T, H, W, C) uint8 [0, 255]
    returns: (1, C, T, H, W) float32
    """
    mean = np.array(cfg.DATA.MEAN, dtype=np.float32)
    std = np.array(cfg.DATA.STD, dtype=np.float32)
    
    frames = frames.astype(np.float32) / 255.0
    frames = (frames - mean) / std
    
    # Transpose to (C, T, H, W) and add batch dim
    frames = np.transpose(frames, (3, 0, 1, 2))
    frames = np.expand_dims(frames, axis=0)
    return torch.from_numpy(frames).float()


def test_with_real_data(pytorch_model, onnx_path, test_path, is_video, cfg, class_names):
    """Test inference on real data."""
    print("\n" + "="*70)
    print("Testing with Real Data")
    print("="*70)
    print(f"Input: {test_path}")
    print(f"Type: {'Video file' if is_video else 'Frames directory'}")
    
    num_frames = cfg.DATA.NUM_FRAMES
    sampling_rate = cfg.DATA.SAMPLING_RATE
    crop_size = cfg.DATA.TEST_CROP_SIZE
    
    # Load frames
    print(f"\nLoading {num_frames} frames (size: {crop_size}x{crop_size})...")
    if is_video:
        frames = load_video_frames(test_path, num_frames, sampling_rate, crop_size)
    else:
        frames = load_frames_from_directory(test_path, num_frames, sampling_rate, crop_size)
    
    print(f"Loaded: {frames.shape}")
    input_tensor = preprocess_frames(frames, cfg)
    print(f"Preprocessed: {input_tensor.shape}")
    
    # PyTorch inference
    print("\n" + "-"*70)
    print("PyTorch Model Inference:")
    print("-"*70)
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_out = pytorch_model([input_tensor])
        if isinstance(pytorch_out, (list, tuple)):
            pytorch_out = pytorch_out[0]
    
    pytorch_logits = pytorch_out.numpy()[0]
    pytorch_probs = np.exp(pytorch_logits - np.max(pytorch_logits))
    pytorch_probs /= pytorch_probs.sum()
    pytorch_class = np.argmax(pytorch_logits)
    pytorch_conf = pytorch_probs[pytorch_class]
    
    print(f"Logits:  {pytorch_logits}")
    print(f"Probs:   {pytorch_probs}")
    print(f"Class:   {pytorch_class}", end="")
    if class_names and pytorch_class < len(class_names):
        print(f" ({class_names[pytorch_class]})")
    else:
        print()
    print(f"Confidence: {pytorch_conf:.4f} ({pytorch_conf*100:.2f}%)")
    
    # ONNX inference
    print("\n" + "-"*70)
    print("ONNX Model Inference:")
    print("-"*70)
    session = ort.InferenceSession(onnx_path)
    onnx_out = session.run(None, {session.get_inputs()[0].name: input_tensor.numpy()})[0]
    
    onnx_logits = onnx_out[0]
    onnx_probs = np.exp(onnx_logits - np.max(onnx_logits))
    onnx_probs /= onnx_probs.sum()
    onnx_class = np.argmax(onnx_logits)
    onnx_conf = onnx_probs[onnx_class]
    
    print(f"Logits:  {onnx_logits}")
    print(f"Probs:   {onnx_probs}")
    print(f"Class:   {onnx_class}", end="")
    if class_names and onnx_class < len(class_names):
        print(f" ({class_names[onnx_class]})")
    else:
        print()
    print(f"Confidence: {onnx_conf:.4f} ({onnx_conf*100:.2f}%)")
    
    # Compare
    print("\n" + "-"*70)
    print("Comparison:")
    print("-"*70)
    diff = np.abs(pytorch_logits - onnx_logits)
    print(f"Max difference:  {diff.max():.8f}")
    print(f"Mean difference: {diff.mean():.8f}")
    
    if pytorch_class == onnx_class:
        print(f"\n✓ Both predict class {pytorch_class}", end="")
        if class_names and pytorch_class < len(class_names):
            print(f" ({class_names[pytorch_class]})")
        else:
            print()
        print(f"  PyTorch: {pytorch_conf:.4f}")
        print(f"  ONNX:    {onnx_conf:.4f}")
        print(f"  Diff:    {abs(pytorch_conf - onnx_conf):.4f}")
    else:
        print("\n✗ Different predictions!")
        print(f"  PyTorch: {pytorch_class} (conf: {pytorch_conf:.4f})")
        print(f"  ONNX:    {onnx_class} (conf: {onnx_conf:.4f})")
    
    print("="*70 + "\n")


def main():
    args = parse_args()
    
    print("="*70)
    print("X3D ONNX Export with Real Data Testing")
    print("="*70)
    
    # Load config
    print(f"\nLoading config from: {args.cfg}")
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)
    cfg.NUM_GPUS = 0
    
    # Build model
    print("Building model...")
    model = build_model(cfg)
    load_checkpoint(args.checkpoint, model, False, None, False, False)
    model = model.cpu().eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded successfully")
    print(f"  Total parameters: {total_params:,}")
    
    # Prepare input
    batch_size = 1
    channels = 3
    num_frames = cfg.DATA.NUM_FRAMES
    height = cfg.DATA.TEST_CROP_SIZE
    width = cfg.DATA.TEST_CROP_SIZE
    
    print(f"\nDummy input shape: ({batch_size}, {channels}, {num_frames}, {height}, {width})")
    dummy_input = torch.randn(batch_size, channels, num_frames, height, width)
    
    # Wrapper for original model (outputs logits)
    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            o = self.m([x])
            return o[0] if isinstance(o, (list, tuple)) else o
    
    wrapped = Wrapper(model).eval()
    
    # Wrapper with softmax for inference
    class WrapperWithSoftmax(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            o = self.m([x])
            logits = o[0] if isinstance(o, (list, tuple)) else o
            return torch.nn.functional.softmax(logits, dim=1)
    
    wrapped_with_softmax = WrapperWithSoftmax(model).eval()
    
    # Export ONNX (original model with logits)
    print(f"\nExporting original model (logits output) to ONNX: {args.output}")
    print(f"Opset version: {args.opset}")
    print(f"Using legacy exporter: {not args.use_dynamo}")
    
    try:
        torch.onnx.export(
            wrapped,
            dummy_input,
            args.output,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            verbose=False,
            dynamo=args.use_dynamo,
        )
        print(f"✓ Exported original model to: {args.output}")
    except Exception as e:
        print(f"✗ Export failed: {e}")
        return
    
    # Simplify if requested
    onnx_path = args.output
    if args.simplify:
        print("\nSimplifying ONNX model...")
        simplified_path = args.output.replace(".onnx", "_simplified.onnx")
        success = simplify_onnx_model(args.output, simplified_path)
        if success:
            onnx_path = simplified_path
    
    # Test with dummy data
    print(f"\n{'='*70}")
    print("Testing Inference Accuracy (Dummy Data)")
    print(f"{'='*70}")
    
    with torch.no_grad():
        pytorch_output = wrapped(dummy_input)
    
    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(None, {session.get_inputs()[0].name: dummy_input.numpy()})[0]
    
    pytorch_np = pytorch_output.detach().numpy()
    
    print(f"\nPyTorch output shape: {pytorch_np.shape}")
    print(f"ONNX output shape:    {onnx_output.shape}")
    print(f"\nPyTorch output sample: {pytorch_np[0]}")
    print(f"ONNX output sample:    {onnx_output[0]}")
    
    diff = np.abs(pytorch_np - onnx_output)
    print(f"\nMax absolute difference:  {diff.max():.8f}")
    print(f"Mean absolute difference: {diff.mean():.8f}")
    
    threshold = 1e-5
    if diff.max() < threshold:
        print(f"✓ Outputs match within threshold ({threshold})")
    else:
        print(f"✗ Outputs differ more than threshold ({threshold})")
        if args.use_dynamo:
            print(f"  Try using --no-use_dynamo for legacy exporter")
    
    print(f"{'='*70}\n")
    
    # Export model with softmax if requested
    softmax_onnx_path = None
    if args.export_softmax:
        softmax_output = args.output.replace(".onnx", "_softmax.onnx")
        print(f"{'='*70}")
        print("Exporting Model with Softmax Layer")
        print(f"{'='*70}")
        print(f"\nExporting model with softmax to: {softmax_output}")
        
        try:
            torch.onnx.export(
                wrapped_with_softmax,
                dummy_input,
                softmax_output,
                export_params=True,
                opset_version=args.opset,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['probabilities'],
                verbose=False,
                dynamo=args.use_dynamo,
            )
            print(f"✓ Exported model with softmax to: {softmax_output}")
            softmax_onnx_path = softmax_output
            
            # Test softmax model output
            print(f"\nTesting Softmax Model Output (Dummy Data)")
            print("-" * 70)
            
            with torch.no_grad():
                pytorch_softmax_output = wrapped_with_softmax(dummy_input)
            
            session_softmax = ort.InferenceSession(softmax_output)
            onnx_softmax_output = session_softmax.run(
                None, 
                {session_softmax.get_inputs()[0].name: dummy_input.numpy()}
            )[0]
            
            pytorch_softmax_np = pytorch_softmax_output.detach().numpy()
            
            print(f"PyTorch softmax output shape: {pytorch_softmax_np.shape}")
            print(f"ONNX softmax output shape:    {onnx_softmax_output.shape}")
            print(f"\nPyTorch softmax sample: {pytorch_softmax_np[0]}")
            print(f"ONNX softmax sample:    {onnx_softmax_output[0]}")
            print(f"\nSum of probabilities (should be ~1.0):")
            print(f"  PyTorch: {pytorch_softmax_np[0].sum():.8f}")
            print(f"  ONNX:    {onnx_softmax_output[0].sum():.8f}")
            
            diff_softmax = np.abs(pytorch_softmax_np - onnx_softmax_output)
            print(f"\nMax absolute difference:  {diff_softmax.max():.8f}")
            print(f"Mean absolute difference: {diff_softmax.mean():.8f}")
            
            if diff_softmax.max() < threshold:
                print(f"✓ Softmax outputs match within threshold ({threshold})")
            else:
                print(f"⚠ Softmax outputs differ more than threshold ({threshold})")
            
            # Simplify softmax model if requested
            if args.simplify:
                print(f"\nSimplifying softmax ONNX model...")
                simplified_softmax_path = softmax_output.replace(".onnx", "_simplified.onnx")
                success = simplify_onnx_model(softmax_output, simplified_softmax_path)
                if success:
                    softmax_onnx_path = simplified_softmax_path
            
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"✗ Softmax export failed: {e}")
    
    # Test with real data if provided
    if args.test_video or args.test_frames:
        class_names = None
        if args.class_names:
            class_names = [s.strip() for s in args.class_names.split(',')]
        
        if args.test_video:
            test_with_real_data(model, onnx_path, args.test_video, True, cfg, class_names)
        
        if args.test_frames:
            test_with_real_data(model, onnx_path, args.test_frames, False, cfg, class_names)
    
    print("✓ Export completed successfully!")
    print(f"  Original model (logits): {onnx_path}")
    if softmax_onnx_path:
        print(f"  Model with softmax:      {softmax_onnx_path}")


if __name__ == "__main__":
    main()
