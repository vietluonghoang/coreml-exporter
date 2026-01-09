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

# Try to import coremltools ONNX converter directly
try:
    from coremltools.converters import onnx as ct_onnx
    CT_ONNX_CONVERTER_AVAILABLE = True
except ImportError:
    CT_ONNX_CONVERTER_AVAILABLE = False


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
    print("\nConverting ONNX to CoreML using coremltools...")
    
    ml_model = None
    
    # Method 1: Use coremltools.converters.onnx directly (specialized ONNX converter)
    if CT_ONNX_CONVERTER_AVAILABLE and ONNX_AVAILABLE:
        print("Method 1: Using coremltools.converters.onnx (specialized)...")
        try:
            # Load ONNX model without validation
            onnx_model = onnx.load(onnx_path)
            
            # Use specialized ONNX converter
            ml_model = ct_onnx.convert(onnx_model)
            print("✓ Converted successfully with ct.converters.onnx")
            
        except Exception as e:
            print(f"⚠ ct.converters.onnx failed: {str(e)[:100]}")
            ml_model = None
    
    # Method 2: Try mlprogram with source="onnx" (if supported)
    if ml_model is None:
        print("Method 2: Attempting mlprogram with explicit source...")
        try:
            ml_model = ct.convert(
                onnx_path,
                convert_to="mlprogram",
                source="onnx"
            )
            print("✓ Converted successfully with mlprogram")
            
        except Exception as e:
            print(f"⚠ mlprogram with source='onnx' failed: {str(e)[:100]}")
    
    # Method 3: neuralnetwork format (fallback)
    if ml_model is None:
        print("Method 3: Attempting neuralnetwork format...")
        try:
            ml_model = ct.convert(
                onnx_path,
                convert_to="neuralnetwork",
                source="onnx"
            )
            print("✓ Converted successfully with neuralnetwork")
            
        except Exception as e:
            print(f"⚠ neuralnetwork also failed: {str(e)[:100]}")
    
    # Method 4: Last resort - load ONNX and pass object to ct.convert
    if ml_model is None and ONNX_AVAILABLE:
        print("Method 4: Direct model object conversion...")
        try:
            onnx_model = onnx.load(onnx_path)
            
            # Try passing model object without source
            ml_model = ct.convert(
                onnx_model,
                convert_to="neuralnetwork"
            )
            print("✓ Converted successfully with model object")
            
        except Exception as e:
            print(f"✗ All conversion methods failed: {str(e)[:150]}")
            print("\nThis ONNX IR v10 + MobileNetV3 model requires PyTorch export:")
            print("1. Export original PyTorch model to .pt format")
            print("2. Place .pt file in project root")
            print("3. Run: python scripts/export_pytorch_to_coreml.py")
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
