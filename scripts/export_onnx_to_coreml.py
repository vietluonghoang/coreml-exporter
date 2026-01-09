#!/usr/bin/env python3
"""
Convert ONNX model to CoreML format
MobileNetV3Small: Input shape (1, 3, 128, 128)
Uses onnx-coreml package (dedicated ONNX converter)
"""

import os
import sys
from pathlib import Path

from onnx_coreml import convert
import onnx


def export_onnx_to_coreml(onnx_path: str, output_dir: str = "output") -> str:
    """
    Convert ONNX model to CoreML format using onnx-coreml.
    
    Args:
        onnx_path: Path to the ONNX model file
        output_dir: Directory to save the CoreML model
        
    Returns:
        Path to the generated CoreML model
    """
    # Load ONNX model
    print(f"Loading ONNX model from: {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    print(f"✓ ONNX model loaded (IR version: {onnx_model.ir_version})")
    
    # Get model info
    producer_name = onnx_model.producer_name
    print(f"ONNX Producer: {producer_name}")
    
    # Extract input info
    input_name = onnx_model.graph.input[0].name
    input_shape = tuple(
        dim.dim_value
        for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim
    )
    print(f"Input: {input_name}, Shape: {input_shape}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to CoreML using onnx-coreml (stable, dedicated package)
    print("Converting ONNX to CoreML using onnx-coreml...")
    try:
        ml_model = convert(onnx_model)
        print("✓ Converted successfully")
        
    except Exception as e:
        print(f"Conversion failed: {str(e)}")
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
