#!/usr/bin/env python3
"""
Convert ONNX model to CoreML format
MobileNetV3Small: Input shape (1, 3, 128, 128)
Uses coremltools 8.3+ which supports ONNX IR v10
Strategy: Optimize ONNX model with onnx-simplifier first
"""

import os
import sys
from pathlib import Path

import onnx
import coremltools as ct

try:
    from onnxsim import simplify
    ONNX_SIMPLIFIER_AVAILABLE = True
except ImportError:
    ONNX_SIMPLIFIER_AVAILABLE = False
    print("Warning: onnx-simplifier not available, skipping optimization")


def export_onnx_to_coreml(onnx_path: str, output_dir: str = "output") -> str:
    """
    Convert ONNX model to CoreML format using coremltools 8.3+
    
    Args:
        onnx_path: Path to the ONNX model file
        output_dir: Directory to save the CoreML model
        
    Returns:
        Path to the generated CoreML model
        
    Strategy:
    1. Load and validate ONNX model (IR v10 support)
    2. Optimize with onnx-simplifier (reduce complexity, improve conversion)
    3. Convert directly with coremltools 8.3+ (native ONNX IR v10 support)
    """
    # Load ONNX model
    print(f"Loading ONNX model from: {onnx_path}")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    
    onnx_model = onnx.load(onnx_path)
    print(f"✓ ONNX model loaded (IR version: {onnx_model.ir_version})")
    
    # Validate model structure
    try:
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validation passed")
    except onnx.checker.ValidationError as e:
        print(f"⚠ ONNX model validation warning: {e}")
    
    # Get model info
    producer_name = onnx_model.producer_name or "Unknown"
    print(f"ONNX Producer: {producer_name}")
    
    # Extract input info
    try:
        input_name = onnx_model.graph.input[0].name
        input_shape = tuple(
            dim.dim_value
            for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim
        )
        print(f"Input: {input_name}, Shape: {input_shape}")
    except (IndexError, AttributeError) as e:
        print(f"Warning: Could not extract input info: {e}")
    
    # Step 1: Optimize ONNX model with onnx-simplifier
    print("\nOptimizing ONNX model...")
    if ONNX_SIMPLIFIER_AVAILABLE:
        try:
            onnx_model, check_ok = simplify(onnx_model)
            if check_ok:
                print("✓ ONNX model optimized successfully")
            else:
                print("⚠ ONNX simplification succeeded but model check failed (continuing anyway)")
        except Exception as e:
            print(f"⚠ ONNX simplification failed: {e} (continuing without optimization)")
    else:
        print("⚠ onnx-simplifier not available, skipping optimization")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 2: Convert to CoreML using coremltools 8.3+
    print("\nConverting ONNX to CoreML using coremltools 8.3+...")
    try:
        ml_model = ct.convert(
            onnx_model,
            convert_to="mlprogram",  # mlprogram = Neural Networks 5 (more modern)
            minimum_ios_deployment_target="14"
        )
        print("✓ Converted successfully with coremltools")
        
    except Exception as e:
        print(f"✗ Conversion failed: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Model may contain unsupported operators")
        print("2. Try 'neuralnetwork' format instead of 'mlprogram'")
        print("3. Export PyTorch model to .pt and convert directly")
        raise
    
    # Determine output filename
    model_name = Path(onnx_path).stem
    output_path = os.path.join(output_dir, f"{model_name}.mlmodel")
    
    # Save CoreML model
    print(f"Saving CoreML model to: {output_path}")
    ml_model.save(output_path)
    print(f"✓ CoreML model saved successfully")
    print(f"✓ Output: {output_path}")
    
    return output_path


def main():
    """Main entry point"""
    # Find ONNX model
    onnx_dir = Path("onnx")
    onnx_files = list(onnx_dir.glob("*.onnx"))
    
    if not onnx_files:
        print("Error: No ONNX files found in 'onnx' directory")
        sys.exit(1)
    
    onnx_model_path = str(onnx_files[0])
    print(f"Found ONNX model: {onnx_model_path}\n")
    
    try:
        export_onnx_to_coreml(onnx_model_path)
        print("\n✓ Conversion completed successfully!")
    except Exception as e:
        print(f"\n✗ Conversion failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
