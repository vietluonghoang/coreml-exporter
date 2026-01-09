#!/usr/bin/env python3
"""
Convert ONNX model to CoreML format
MobileNetV3Small: Input shape (1, 3, 128, 128)
Uses onnx-coreml package (dedicated ONNX converter)
Fallback: coremltools with direct ONNX support
"""

import os
import sys
from pathlib import Path

import onnx
try:
    from onnx_coreml import convert as onnx_coreml_convert
    ONNX_COREML_AVAILABLE = True
except ImportError:
    ONNX_COREML_AVAILABLE = False
    print("Warning: onnx-coreml not available, will try coremltools fallback")


def export_onnx_to_coreml(onnx_path: str, output_dir: str = "output") -> str:
    """
    Convert ONNX model to CoreML format using onnx-coreml.
    
    Args:
        onnx_path: Path to the ONNX model file
        output_dir: Directory to save the CoreML model
        
    Returns:
        Path to the generated CoreML model
        
    Strategy: Use onnx-coreml 1.3+ (specialized ONNX converter)
    - Load model as Python object (NOT file path)
    - Pass to convert() without 'source' parameter
    - Supports ONNX IR v10 with MobileNetV3 HardSwish operators
    """
    # Load ONNX model
    print(f"Loading ONNX model from: {onnx_path}")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)  # Validate model structure
    print(f"✓ ONNX model loaded (IR version: {onnx_model.ir_version})")
    
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
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to CoreML
    ml_model = None
    
    # Method 1: Try onnx-coreml (preferred, specialized ONNX converter)
    if ONNX_COREML_AVAILABLE:
        print("Converting ONNX to CoreML using onnx-coreml...")
        try:
            ml_model = onnx_coreml_convert(
                onnx_model,
                # ✅ CORRECT: Pass model object, NOT file path
                # ❌ DO NOT use source="pytorch" - causes error with ONNX objects
            )
            print("✓ Converted successfully with onnx-coreml")
        except Exception as e:
            print(f"✗ onnx-coreml conversion failed: {str(e)}")
            print("Attempting fallback method...")
            ml_model = None
    
    # Method 2: Fallback to coremltools with flexible_shape_ranges
    if ml_model is None:
        print("Converting ONNX to CoreML using coremltools...")
        try:
            import coremltools as ct
            
            # coremltools 7.x can handle ONNX files directly
            ml_model = ct.converters.onnx.convert(
                onnx_model,
                minimum_ios_deployment_target="14"
            )
            print("✓ Converted successfully with coremltools fallback")
        except Exception as e:
            print(f"✗ coremltools conversion also failed: {str(e)}")
            print("\nFinal troubleshooting:")
            print("1. Check ONNX model validity: onnx.checker.check_model()")
            print("2. Model may have unsupported operators for CoreML")
            print("3. Consider optimizing ONNX model with onnx-simplifier")
            print("4. Last resort: Export PyTorch model to .pt format first")
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
