#!/usr/bin/env python3
"""
Convert ONNX model to CoreML format
MobileNetV3Small: Input shape (1, 3, 128, 128)
"""

import os
import sys
from pathlib import Path

import coremltools as ct
import onnx
from onnx import version_converter


def export_onnx_to_coreml(onnx_path: str, output_dir: str = "output") -> str:
    """
    Convert ONNX model to CoreML format.
    
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
    
    # Downgrade IR version if needed (coremltools 7.2 supports up to IR v9)
    if onnx_model.ir_version > 9:
        print(f"⚠ Downgrading ONNX IR from v{onnx_model.ir_version} to v9 (coremltools 7.2 compatibility)")
        onnx_model = version_converter.convert_version(onnx_model, 9)
        temp_onnx_path = os.path.join(output_dir, "model_ir9.onnx")
        onnx.save(onnx_model, temp_onnx_path)
        onnx_path = temp_onnx_path
        print(f"✓ Downgraded to IR v9")
    
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
    
    # Convert to CoreML with source parameter
    print("Converting ONNX to CoreML...")
    try:
        ml_model = ct.convert(
            onnx_path,
            source="pytorch",
            convert_to="neuralnetwork"
        )
        print("✓ Converted to Neural Network successfully")
        
    except Exception as e:
        print(f"Neural Network conversion failed: {str(e)[:200]}")
        print("Retrying with mlprogram...")
        try:
            ml_model = ct.convert(
                onnx_path,
                source="pytorch",
                convert_to="mlprogram"
            )
            print("✓ Converted to ML Program successfully")
        except Exception as e2:
            print(f"ML Program conversion also failed: {str(e2)[:200]}")
            raise
    
    # Determine output filename
    model_name = Path(onnx_path).stem.replace("_ir9", "")
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
