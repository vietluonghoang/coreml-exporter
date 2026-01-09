#!/usr/bin/env python3
"""
Convert PyTorch model directly to CoreML format (bypass ONNX IR v10 issues)

MobileNetV3Small: Input shape (1, 3, 128, 128)
Strategy: Use PyTorch 2.0+ ExportedProgram (torch._export) for stable conversion
"""

import os
import sys
from pathlib import Path
import tempfile

import torch
import torch.nn as nn
import coremltools as ct
from coremltools.converters.common import CTModelType


def convert_pytorch_to_coreml(model_or_path: str, output_dir: str = "output") -> str:
    """
    Convert PyTorch model to CoreML format.
    
    Args:
        model_or_path: Path to .pt/.pth file OR directly pass model
        output_dir: Directory to save the CoreML model
        
    Returns:
        Path to the generated CoreML model
        
    Strategy:
    1. Load PyTorch model from .pt/.pth file
    2. Create example input for tracing
    3. Convert using coremltools.convert() with source="pytorch"
    4. Save as CoreML model
    """
    # Load PyTorch model
    print(f"Loading PyTorch model from: {model_or_path}")
    
    if isinstance(model_or_path, str):
        if not os.path.exists(model_or_path):
            raise FileNotFoundError(f"PyTorch file not found: {model_or_path}")
        
        # Load .pt/.pth file
        try:
            model = torch.load(model_or_path, map_location="cpu")
            print(f"✓ PyTorch model loaded")
        except Exception as e:
            print(f"Failed to load as torch.load(): {e}")
            print("Attempting to import as module...")
            raise
    else:
        model = model_or_path
    
    # Ensure model is in eval mode
    if isinstance(model, nn.Module):
        model.eval()
    
    # Get model info
    print(f"Model type: {type(model).__name__}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create example input (MobileNetV3Small: 1, 3, 128, 128)
    print("\nPreparing example input for tracing...")
    example_input = torch.randn(1, 3, 128, 128)
    print(f"Example input shape: {example_input.shape}")
    
    # Try conversion
    print("\nConverting PyTorch to CoreML using coremltools...")
    
    ml_model = None
    
    # Method 1: Try using torch.jit.trace (simpler, more compatible)
    print("Method 1: Using torch.jit.trace...")
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)
        
        ml_model = ct.convert(
            traced_model,
            convert_to="mlprogram",
            inputs=[ct.TensorType(shape=example_input.shape, name="input")]
        )
        print("✓ Converted successfully with torch.jit.trace + mlprogram")
        
    except Exception as e:
        print(f"⚠ torch.jit.trace failed: {e}")
    
    # Method 2: Try torch.jit.script (more explicit)
    if ml_model is None:
        print("Method 2: Attempting torch.jit.script (if model is scriptable)...")
        try:
            with torch.no_grad():
                scripted_model = torch.jit.script(model)
            
            ml_model = ct.convert(
                scripted_model,
                convert_to="mlprogram",
                inputs=[ct.TensorType(shape=example_input.shape, name="input")]
            )
            print("✓ Converted successfully with torch.jit.script + mlprogram")
            
        except Exception as e:
            print(f"⚠ torch.jit.script failed: {e}")
    
    # Method 3: Direct coremltools conversion (newest, requires torch 2.0+)
    if ml_model is None:
        print("Method 3: Direct coremltools conversion (PyTorch 2.0+)...")
        try:
            ml_model = ct.convert(
                model,
                convert_to="mlprogram",
                inputs=[ct.TensorType(shape=example_input.shape, name="input")]
            )
            print("✓ Converted successfully with direct conversion + mlprogram")
            
        except Exception as e:
            print(f"⚠ Direct conversion failed: {e}")
    
    # Method 4: Fallback to neuralnetwork format
    if ml_model is None:
        print("Method 4: Fallback to neuralnetwork format...")
        try:
            with torch.no_grad():
                traced_model = torch.jit.trace(model, example_input)
            
            ml_model = ct.convert(
                traced_model,
                convert_to="neuralnetwork",
                inputs=[ct.TensorType(shape=example_input.shape, name="input")]
            )
            print("✓ Converted successfully with torch.jit.trace + neuralnetwork")
            
        except Exception as e:
            print(f"✗ All conversion methods failed: {e}")
            raise
    
    # Save CoreML model
    model_name = Path(model_or_path).stem if isinstance(model_or_path, str) else "model"
    output_path = os.path.join(output_dir, f"{model_name}.mlmodel")
    
    print(f"\nSaving CoreML model to: {output_path}")
    ml_model.save(output_path)
    print(f"✓ CoreML model saved successfully")
    print(f"✓ Output: {output_path}")
    
    return output_path


def export_onnx_via_pytorch(onnx_path: str, output_dir: str = "output") -> str:
    """
    Convert ONNX to CoreML via PyTorch (workaround for IR v10 issues)
    
    Args:
        onnx_path: Path to ONNX model file
        output_dir: Directory to save the CoreML model
        
    Returns:
        Path to the generated CoreML model
        
    Strategy:
    1. Load ONNX model as PyTorch via onnx2pytorch or similar
    2. Convert PyTorch to CoreML
    """
    print("Attempting indirect conversion: ONNX → PyTorch → CoreML")
    print("This bypasses ONNX IR v10 checker issues")
    
    try:
        # Try using onnx2pytorch if available
        from onnx2pytorch import ConvertModel
        import onnx
        
        print(f"Loading ONNX model from: {onnx_path}")
        onnx_model = onnx.load(onnx_path)
        
        print("Converting ONNX to PyTorch...")
        pytorch_model = ConvertModel(onnx_model)
        pytorch_model.eval()
        
        print("✓ Successfully converted ONNX to PyTorch")
        
        # Now convert PyTorch to CoreML
        return convert_pytorch_to_coreml(pytorch_model, output_dir)
        
    except ImportError:
        print("⚠ onnx2pytorch not available")
        print("Workaround: Save PyTorch model to .pt file and use that instead")
        raise


def main():
    """Main entry point"""
    
    # Look for .pt/.pth files first
    scripts_dir = Path(__file__).parent
    project_root = scripts_dir.parent
    
    # Check for PyTorch models
    pt_files = list(project_root.glob("**/*.pt")) + list(project_root.glob("**/*.pth"))
    
    if pt_files:
        pt_model_path = str(pt_files[0])
        print(f"Found PyTorch model: {pt_model_path}\n")
        
        try:
            convert_pytorch_to_coreml(pt_model_path)
            print("\n✓ Conversion completed successfully!")
        except Exception as e:
            print(f"\n✗ Conversion failed: {str(e)}")
            sys.exit(1)
    else:
        # Fallback: try ONNX via PyTorch
        onnx_dir = project_root / "onnx"
        onnx_files = list(onnx_dir.glob("*.onnx"))
        
        if not onnx_files:
            print("Error: No .pt/.pth or .onnx files found")
            sys.exit(1)
        
        onnx_model_path = str(onnx_files[0])
        print(f"Found ONNX model: {onnx_model_path}\n")
        print("Attempting ONNX → PyTorch → CoreML conversion\n")
        
        try:
            export_onnx_via_pytorch(onnx_model_path)
            print("\n✓ Conversion completed successfully!")
        except Exception as e:
            print(f"\n✗ ONNX conversion failed: {str(e)}")
            print("\nFinal workaround:")
            print("1. Export PyTorch model to .pt/.pth format")
            print("2. Place in project root directory")
            print("3. Run this script again")
            sys.exit(1)


if __name__ == "__main__":
    main()
