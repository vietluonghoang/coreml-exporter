#!/usr/bin/env python3
"""
Convert ONNX model to CoreML format
"""

import os
import sys
from pathlib import Path

import coremltools as ct
import onnx
import numpy as np


def get_input_shapes(onnx_path: str) -> dict:
    """
    Extract input shapes from ONNX model.
    """
    onnx_model = onnx.load(onnx_path)
    input_shapes = {}
    
    for input_info in onnx_model.graph.input:
        shape = []
        for dim in input_info.type.tensor_type.shape.dim:
            if dim.dim_value == 0:
                shape.append(1)  # Default to 1 if dynamic
            else:
                shape.append(dim.dim_value)
        input_shapes[input_info.name] = shape
    
    return input_shapes


def detect_source_framework(onnx_path: str) -> str:
    """
    Detect the source framework from ONNX model.
    Defaults to pytorch for most models.
    """
    onnx_model = onnx.load(onnx_path)
    producer_name = onnx_model.producer_name.lower()
    
    print(f"ONNX Producer: {producer_name}")
    
    if "pytorch" in producer_name:
        return "pytorch"
    elif "tensorflow" in producer_name:
        return "tensorflow"
    else:
        # Default to pytorch
        print("Defaulting to pytorch source framework")
        return "pytorch"


def export_onnx_to_coreml(onnx_path: str, output_dir: str = "output") -> str:
    """
    Convert ONNX model to CoreML format.
    
    Args:
        onnx_path: Path to the ONNX model file
        output_dir: Directory to save the CoreML model
        
    Returns:
        Path to the generated CoreML model
    """
    # Validate ONNX model
    print(f"Loading ONNX model from: {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model validation passed")
    
    # Get input shapes
    input_shapes = get_input_shapes(onnx_path)
    print(f"Input shapes: {input_shapes}")
    
    # Detect source framework
    source_framework = detect_source_framework(onnx_path)
    print(f"Source framework detected: {source_framework}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to CoreML
    print("Converting ONNX to CoreML...")
    try:
        ml_model = ct.convert(
            onnx_path,
            convert_to="neuralnetwork"
        )
    except Exception as e:
        print(f"Neuralnetwork conversion failed: {e}")
        print("Retrying with mlprogram...")
        ml_model = ct.convert(
            onnx_path,
            convert_to="mlprogram"
        )
    
    # Determine output filename
    model_name = Path(onnx_path).stem
    output_path = os.path.join(output_dir, f"{model_name}.mlpackage")
    
    # Save CoreML model
    print(f"Saving CoreML model to: {output_path}")
    ml_model.save(output_path)
    print(f"✓ CoreML model saved successfully")
    
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
    print(f"Found ONNX model: {onnx_model_path}")
    
    try:
        output_path = export_onnx_to_coreml(onnx_model_path)
        print(f"\n✓ Conversion completed successfully!")
        print(f"Output: {output_path}")
    except Exception as e:
        print(f"\n✗ Conversion failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
