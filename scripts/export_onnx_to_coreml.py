#!/usr/bin/env python3
"""
Convert ONNX model to CoreML format
MobileNetV3Small: Input shape (1, 3, 128, 128)

CRITICAL: ONNX IR v10 + MobileNetV3 HardSwish causes segfault in coremltools
Strategy: Skip ONNX validation, use low-level coremltools API directly
"""

import os
import sys
from pathlib import Path
import warnings

# Suppress validation warnings
warnings.filterwarnings("ignore")

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

import coremltools as ct


def export_onnx_to_coreml(onnx_path: str, output_dir: str = "output") -> str:
    """
    Convert ONNX model to CoreML format using coremltools.
    
    CRITICAL FIX for ONNX IR v10 + MobileNetV3 segfault:
    - Load ONNX WITHOUT validation (skip onnx.checker)
    - Pass file path directly to ct.convert (not model object)
    - Let coremltools handle format detection
    
    Args:
        onnx_path: Path to the ONNX model file
        output_dir: Directory to save the CoreML model
        
    Returns:
        Path to the generated CoreML model
    """
    # Verify file exists
    print(f"Loading ONNX model from: {onnx_path}")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    
    # Load ONNX for info only (NO validation to avoid IR v10 crash)
    if ONNX_AVAILABLE:
        try:
            onnx_model = onnx.load(onnx_path)
            print(f"✓ ONNX model loaded (IR version: {onnx_model.ir_version})")
            
            producer_name = onnx_model.producer_name or "Unknown"
            print(f"ONNX Producer: {producer_name}")
            
            # Extract input info (no validation)
            try:
                input_name = onnx_model.graph.input[0].name
                input_shape = tuple(
                    dim.dim_value
                    for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim
                )
                print(f"Input: {input_name}, Shape: {input_shape}")
            except (IndexError, AttributeError):
                pass
        except Exception as e:
            print(f"⚠ ONNX info loading skipped: {e}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to CoreML
    # CRITICAL: Pass file path string, not model object
    # Let coremltools 8.3+ handle ONNX detection internally (no segfault)
    print("\nConverting ONNX to CoreML using coremltools...")
    
    ml_model = None
    
    # Method 1: mlprogram (modern)
    print("Method 1: Attempting mlprogram format...")
    try:
        # Pass file path directly - coremltools will detect and load internally
        ml_model = ct.convert(
            onnx_path,
            convert_to="mlprogram"
        )
        print("✓ Converted successfully with mlprogram")
        
    except Exception as e:
        print(f"⚠ mlprogram failed: {str(e)[:100]}")
    
    # Method 2: neuralnetwork (fallback)
    if ml_model is None:
        print("Method 2: Attempting neuralnetwork format...")
        try:
            ml_model = ct.convert(
                onnx_path,
                convert_to="neuralnetwork"
            )
            print("✓ Converted successfully with neuralnetwork")
            
        except Exception as e:
            print(f"✗ neuralnetwork also failed: {str(e)[:100]}")
            print("\nNote: This ONNX IR v10 model may need:")
            print("1. Export from PyTorch to .pt format instead")
            print("2. Operator optimization via onnx-simplifier")
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
