#!/usr/bin/env python3
"""
Example: Using exported ONNX model for inference
"""

import numpy as np
import onnxruntime as ort


def load_onnx_model(model_path):
    """Load ONNX model."""
    print(f"Loading ONNX model from: {model_path}")
    session = ort.InferenceSession(model_path)
    
    # Print model info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    
    print(f"\nModel Information:")
    print(f"  Input name: {input_info.name}")
    print(f"  Input shape: {input_info.shape}")
    print(f"  Input type: {input_info.type}")
    print(f"  Output name: {output_info.name}")
    print(f"  Output shape: {output_info.shape}")
    print(f"  Output type: {output_info.type}")
    
    return session


def prepare_video_input(num_frames=32, height=64, width=64, batch_size=1):
    """
    Prepare dummy video input.
    In real application, replace this with actual video preprocessing.
    
    Expected input format: (batch, channel, time, height, width)
    """
    # Generate random input for demonstration
    # In production, this should be your actual video frames
    video_input = np.random.randn(
        batch_size, 3, num_frames, height, width
    ).astype(np.float32)
    
    # Normalize to [0, 1] range (adjust based on your training preprocessing)
    # video_input = (video_input - video_input.min()) / (video_input.max() - video_input.min())
    
    return video_input


def run_inference(session, video_input):
    """Run inference on video input."""
    input_name = session.get_inputs()[0].name
    
    print(f"\nRunning inference...")
    print(f"  Input shape: {video_input.shape}")
    
    # Run inference
    outputs = session.run(None, {input_name: video_input})
    logits = outputs[0]
    
    print(f"  Output shape: {logits.shape}")
    
    return logits


def postprocess_output(logits, class_names=None):
    """Post-process model output."""
    # Get predicted class
    predicted_class = np.argmax(logits, axis=-1)
    
    # Calculate probabilities using softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Get confidence for predicted class
    confidence = probabilities[0, predicted_class[0]]
    
    print(f"\nResults:")
    print(f"  Logits: {logits[0]}")
    print(f"  Probabilities: {probabilities[0]}")
    print(f"  Predicted class: {predicted_class[0]}")
    
    if class_names and predicted_class[0] < len(class_names):
        print(f"  Class name: {class_names[predicted_class[0]]}")
    
    print(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    return predicted_class[0], confidence


def main():
    """Main function."""
    # Configuration
    MODEL_PATH = "output/custom_x3d_3/model.onnx"
    
    # Define your class names (modify according to your dataset)
    CLASS_NAMES = ["Class 0", "Class 1"]  # Update with actual class names
    
    # Model input parameters (should match your training config)
    NUM_FRAMES = 32
    HEIGHT = 64
    WIDTH = 64
    
    print("="*70)
    print("ONNX Model Inference Example")
    print("="*70)
    
    # Load model
    session = load_onnx_model(MODEL_PATH)
    
    # Prepare input (replace with your actual video preprocessing)
    video_input = prepare_video_input(
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        batch_size=1
    )
    
    # Run inference
    logits = run_inference(session, video_input)
    
    # Post-process and display results
    predicted_class, confidence = postprocess_output(logits, CLASS_NAMES)
    
    print("\n" + "="*70)
    print("Inference completed successfully!")
    print("="*70)
    
    # Example: Batch inference
    print("\n\nBatch Inference Example:")
    print("-"*70)
    batch_size = 4
    batch_input = prepare_video_input(
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        batch_size=batch_size
    )
    
    print(f"Processing batch of {batch_size} videos...")
    batch_logits = run_inference(session, batch_input)
    
    for i in range(batch_size):
        pred_class = np.argmax(batch_logits[i])
        exp_logits = np.exp(batch_logits[i] - np.max(batch_logits[i]))
        probs = exp_logits / np.sum(exp_logits)
        confidence = probs[pred_class]
        
        class_name = CLASS_NAMES[pred_class] if pred_class < len(CLASS_NAMES) else f"Class {pred_class}"
        print(f"  Video {i+1}: {class_name} (confidence: {confidence:.4f})")
    
    print("-"*70)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease make sure you have:")
        print("1. Run 'python export_onnx.py' to generate the ONNX model")
        print("2. The model file exists at the specified path")
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install required packages:")
        print("  pip install onnxruntime numpy")
