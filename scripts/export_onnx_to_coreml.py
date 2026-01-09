#!/usr/bin/env python3
"""
Convert ONNX model to CoreML format on GitHub Actions
MobileNetV3Small: Input shape (1, 3, 128, 128)
"""

import os
import sys
from pathlib import Path

import coremltools as ct
import onnx


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

    # Get model info
    producer_name = onnx_model.producer_name
    print(f"ONNX Producer: {producer_name}")

    # Extract input info
    input_name = onnx_model.graph.input[0].name
    input_shape = [
        dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim
    ]
    print(f"Input: {input_name}, Shape: {input_shape}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert to CoreML
    print("Converting ONNX to CoreML...")
    try:
        ml_model = ct.convert(
            onnx_model,
            source="pytorch",
            convert_to="mlprogram",
            inputs=[ct.TensorType(name=input_name, shape=input_shape)],
        )
        print("✓ Converted to ML Program (mlprogram)")

    except Exception as e:
        print(f"ML Program conversion failed: {e}")
        print("Retrying with Neural Network...")
        try:
            ml_model = ct.convert(
                onnx_model,
                source="pytorch",
                convert_to="neuralnetwork",
                inputs=[ct.TensorType(name=input_name, shape=input_shape)],
            )
            print("✓ Converted to Neural Network")
        except Exception as e2:
            print(f"Neural Network conversion also failed: {e2}")
            raise

    # Determine output filename
    model_name = Path(onnx_path).stem
    output_path = os.path.join(output_dir, f"{model_name}.mlpackage")

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
