#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import cv2

try:
    import tensorrt as trt
except ImportError:
    print("TensorRT (tensorrt) is not installed. Please install TensorRT Python package.")
    sys.exit(1)

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 - initializes CUDA context
except ImportError:
    print("PyCUDA is not installed. Install with: pip install pycuda")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="TensorRT inference on a folder of video frames (X3D)")
    parser.add_argument("--engine", required=True, help="Path to TensorRT engine file (.engine)")
    parser.add_argument("--frames", required=True, help="Path to folder containing extracted frames")
    parser.add_argument("--num_frames", type=int, default=32, help="Number of frames per clip (T)")
    parser.add_argument("--sampling_rate", type=int, default=5, help="Temporal sampling rate")
    parser.add_argument("--crop_size", type=int, default=64, help="Spatial crop size (H=W)")
    parser.add_argument("--mean", type=float, nargs=3, default=[0.45, 0.45, 0.45], help="RGB mean for normalization")
    parser.add_argument("--std", type=float, nargs=3, default=[0.225, 0.225, 0.225], help="RGB std for normalization")
    parser.add_argument("--print_input_samples", action="store_true", help="Print first 10 input values and save to python_trt_input.npy")
    parser.add_argument("--input_npy", type=str, default=None, help="Path to .npy file to use as input (overrides frames)")
    parser.add_argument("--class_names", type=str, default=None, help="Comma-separated class names (e.g., 'class0,class1')")
    return parser.parse_args()


def load_engine(engine_path: str):
    if not os.path.isfile(engine_path):
        raise FileNotFoundError(f"Engine file not found: {engine_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine")
    return engine


def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    files.sort()
    return files


def preprocess(frames_dir: str, num_frames: int, sampling_rate: int, crop_size: int, mean, std) -> np.ndarray:
    image_files = list_images(frames_dir)
    if not image_files:
        raise ValueError(f"No images found in {frames_dir}")

    total_frames = len(image_files)
    clip_length = num_frames * sampling_rate
    start_frame = (total_frames - clip_length) // 2 if total_frames > clip_length else 0

    frames = []
    for i in range(num_frames):
        frame_idx = start_frame + i * sampling_rate
        if frame_idx >= total_frames:
            frame_idx = frame_idx % total_frames  # loop pad

        img = cv2.imread(image_files[frame_idx])
        if img is None:
            if frames:
                frames.append(frames[-1])
                continue
            frames.append(np.zeros((crop_size, crop_size, 3), dtype=np.uint8))
            continue

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize short side to crop_size then center crop
        h, w, _ = img.shape
        if h < w:
            new_h = crop_size
            new_w = int(round(w * crop_size / h))
        else:
            new_w = crop_size
            new_h = int(round(h * crop_size / w))
        img = cv2.resize(img, (new_w, new_h))
        start_x = max((new_w - crop_size) // 2, 0)
        start_y = max((new_h - crop_size) // 2, 0)
        img = img[start_y:start_y + crop_size, start_x:start_x + crop_size]

        frames.append(img)

    # T H W C -> normalize -> C T H W -> N C T H W
    tensor = np.stack(frames).astype(np.float32) / 255.0
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    tensor = (tensor - mean) / std
    tensor = np.transpose(tensor, (3, 0, 1, 2))
    tensor = np.expand_dims(tensor, axis=0).astype(np.float32)
    return tensor


def allocate_buffers(engine, context, input_shape, input_dtype):
    # Set input shape for dynamic engine if needed
    input_name = None
    output_name = None
    
    # TensorRT 10.x API: use num_io_tensors instead of num_bindings
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_name = name

    if input_name is None or output_name is None:
        raise RuntimeError("Failed to locate input/output tensors")

    # Dynamic shape handling
    if -1 in tuple(engine.get_tensor_shape(input_name)):
        context.set_input_shape(input_name, tuple(input_shape))

    # Infer output shape from context
    out_shape = tuple(context.get_tensor_shape(output_name))

    # Map TRT dtype to numpy
    def trt_dtype_to_np(dt):
        if dt == trt.DataType.FLOAT:
            return np.float32
        if dt == trt.DataType.HALF:
            return np.float16
        if dt == trt.DataType.INT8:
            return np.int8
        if dt == trt.DataType.INT32:
            return np.int32
        if dt == trt.DataType.BOOL:
            return np.bool_
        raise TypeError(f"Unsupported TRT dtype: {dt}")

    in_dtype_np = trt_dtype_to_np(engine.get_tensor_dtype(input_name))
    out_dtype_np = trt_dtype_to_np(engine.get_tensor_dtype(output_name))

    host_in = np.empty(input_shape, dtype=in_dtype_np)
    host_out = np.empty(out_shape, dtype=out_dtype_np)

    d_in = cuda.mem_alloc(host_in.nbytes)
    d_out = cuda.mem_alloc(host_out.nbytes)

    return input_name, output_name, host_in, host_out, d_in, d_out


def softmax(x: np.ndarray, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def main():
    args = parse_args()

    # Preprocess
    if args.input_npy:
        print(f"Loading input from {args.input_npy}...")
        input_tensor = np.load(args.input_npy)
    else:
        print(f"Preprocessing images from {args.frames}...")
        input_tensor = preprocess(
            args.frames,
            args.num_frames,
            args.sampling_rate,
            args.crop_size,
            args.mean,
            args.std,
        )
    print(f"Input tensor shape: {input_tensor.shape}")

    if args.print_input_samples:
        np.save("python_trt_input.npy", input_tensor.astype(np.float32))
        print("Saved input to python_trt_input.npy")
        print("Sample values:", input_tensor.flatten()[:10])

    # Load engine
    print(f"Loading TensorRT engine from {args.engine}...")
    engine = load_engine(args.engine)
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create execution context")

    # Prepare buffers
    # Engine expects N C T H W
    n, c, t, h, w = input_tensor.shape
    input_shape = (n, c, t, h, w)
    input_dtype = np.float32
    input_name, output_name, host_in, host_out, d_in, d_out = allocate_buffers(
        engine, context, input_shape, input_dtype
    )

    # If engine input is FP16/INT8, cast accordingly
    if host_in.dtype != input_tensor.dtype:
        print(f"Casting input from {input_tensor.dtype} to {host_in.dtype} for engine")
    np.copyto(host_in, input_tensor.astype(host_in.dtype, copy=False))

    # Create CUDA stream
    stream = cuda.Stream()

    # HtoD
    cuda.memcpy_htod_async(d_in, host_in, stream)

    # Set tensor addresses (TensorRT 10.x API)
    context.set_tensor_address(input_name, int(d_in))
    context.set_tensor_address(output_name, int(d_out))

    # Execute
    print("Running inference...")
    ok = context.execute_async_v3(stream.handle)
    if not ok:
        raise RuntimeError("TensorRT execution failed")

    # DtoH
    cuda.memcpy_dtoh_async(host_out, d_out, stream)
    
    # Synchronize stream to ensure all operations complete
    stream.synchronize()

    # Postprocess & print
    output = host_out
    
    # Handle 5D output (N, T, H, W, C) from patched model
    if output.ndim == 5:
        # (N, T, H, W, C) -> Mean on T,H,W
        # Assuming N=1 for inference
        print(f"Output shape: {output.shape} (Probabilities)")
        # Mean over T, H, W (axes 1, 2, 3)
        probs = np.mean(output, axis=(1, 2, 3))
        # Result is (N, C)
    else:
        # Try to infer if output already softmaxed
        probs = output
        if probs.ndim == 2 and not np.allclose(probs.sum(axis=1), 1.0, atol=1e-3):
            probs = softmax(output, axis=1)

    if probs.ndim == 2 and probs.shape[0] == 1:
        probs = probs[0]

    print("Inference Results:")
    if args.class_names:
        classes = [s.strip() for s in args.class_names.split(",")]
        for i, p in enumerate(probs):
            name = classes[i] if i < len(classes) else str(i)
            print(f"{name}: {p:.6f}")
    else:
        for i, p in enumerate(probs):
            print(f"Class {i}: {p:.6f}")

    # Cleanup
    # pycuda.autoinit will clean context on exit


if __name__ == "__main__":
    main()
