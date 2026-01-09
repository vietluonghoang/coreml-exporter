#!/usr/bin/env python3
"""
Convert PyTorch model (.pt) directly to CoreML format

Input: PyTorch model file (.pt)
Output: CoreML model (.mlmodel/.mlpackage)
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import coremltools as ct


def convert_pytorch_to_coreml(pt_path: str, output_dir: str = "output") -> str:
    """
    Convert PyTorch model to CoreML format.
    
    Args:
        pt_path: Path to .pt/.pth file
        output_dir: Directory to save the CoreML model
        
    Returns:
        Path to the generated CoreML model
    """
    # Load PyTorch model
    print(f"Loading PyTorch model from: {pt_path}")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"PyTorch file not found: {pt_path}")
    
    loaded = torch.load(pt_path, map_location="cpu")
    print(f"✓ Loaded checkpoint (type: {type(loaded).__name__})")
    
    model = None
    state_dict = None
    
    # Extract model or state_dict from checkpoint
    if isinstance(loaded, nn.Module):
        model = loaded
        print("✓ Direct nn.Module loaded")
    
    elif isinstance(loaded, dict):
        print(f"Checkpoint keys: {list(loaded.keys())}")
        
        # Try to find model object
        if 'model' in loaded and isinstance(loaded['model'], nn.Module):
            model = loaded['model']
            print("✓ Found nn.Module in checkpoint['model']")
        
        # Look for state_dict (try all common variations)
        for key in ['model_state_dict', 'model', 'state_dict', 'net', 'backbone']:
            if key in loaded and isinstance(loaded[key], dict) and state_dict is None:
                state_dict = loaded[key]
                print(f"Found state_dict in checkpoint['{key}']")
                break
    
    # If only state_dict available, rebuild MobileNetV3Small architecture
    if model is None and state_dict is not None:
        print("\nRebuilding MobileNetV3Small architecture from state_dict...")
        try:
            from torchvision.models import mobilenet_v3_small
            
            # Check if state_dict has 'backbone.' prefix (custom wrapper)
            has_backbone_prefix = any(k.startswith('backbone.') for k in state_dict.keys())
            if has_backbone_prefix:
                print("Detected 'backbone.' prefix in state_dict, stripping...")
                # Remove 'backbone.' prefix to match MobileNetV3Small keys
                state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            
            # Create MobileNetV3Small with random initialization
            model = mobilenet_v3_small(weights=None, num_classes=1000)  # Default num_classes
            print("✓ Created MobileNetV3Small architecture")
            
            # Try to load full state_dict
            try:
                model.load_state_dict(state_dict)
                print("✓ Loaded full state_dict")
            except RuntimeError as e:
                print(f"⚠ Full state_dict loading failed, attempting backbone-only load...")
                
                # Load only backbone (features) part - ignore custom classifier
                backbone_dict = {k.replace('features.', ''): v 
                                 for k, v in state_dict.items() 
                                 if k.startswith('features.')}
                
                if backbone_dict:
                    model.features.load_state_dict(backbone_dict, strict=False)
                    print("✓ Loaded backbone weights (features) only")
                    print("⚠ Note: Classifier head was custom, using default MobileNetV3Small classifier")
                else:
                    print("✗ Could not find 'features' in state_dict")
                    raise e
            
        except ImportError:
            print("✗ torchvision not installed. Install: pip install torchvision")
            raise
        except Exception as e:
            print(f"✗ Failed to load weights: {e}")
            raise
    
    if model is None:
        print("\n✗ Could not load model")
        raise ValueError(
            "Checkpoint format not supported.\n"
            "Need one of:\n"
            "  1. Direct nn.Module object\n"
            "  2. Dict with 'model' key containing nn.Module\n"
            "  3. Dict with 'state_dict' key (will rebuild MobileNetV3Small)"
        )
    
    # Ensure model is in eval mode
    model.eval()
    print(f"✓ Model ready for conversion")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create example input (MobileNetV3Small: 1, 3, 128, 128)
    print("\nPreparing example input for model conversion...")
    example_input = torch.randn(1, 3, 128, 128)
    print(f"Example input shape: {example_input.shape}")
    
    ml_model = None
    
    # Method 1: Direct coremltools conversion (PyTorch 2.0+ native)
    print("\nMethod 1: Direct coremltools conversion...")
    try:
        ml_model = ct.convert(
            model,
            convert_to="mlprogram",
            inputs=[ct.TensorType(shape=example_input.shape, name="input")],
            outputs=[ct.TensorType(name="output")]
        )
        print("✓ Converted successfully with mlprogram format")
        
    except Exception as e:
        print(f"⚠ mlprogram conversion failed: {str(e)[:150]}")
    
    # Method 2: torch.jit.trace + mlprogram
    if ml_model is None:
        print("\nMethod 2: Using torch.jit.trace + mlprogram...")
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
            print(f"⚠ torch.jit.trace + mlprogram failed: {str(e)[:150]}")
    
    # Method 3: torch.jit.trace + neuralnetwork (fallback)
    if ml_model is None:
        print("\nMethod 3: Using torch.jit.trace + neuralnetwork...")
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
            print(f"⚠ torch.jit.trace + neuralnetwork failed: {str(e)[:150]}")
    
    # Method 4: Direct conversion to neuralnetwork (legacy)
    if ml_model is None:
        print("\nMethod 4: Direct neuralnetwork conversion...")
        try:
            ml_model = ct.convert(
                model,
                convert_to="neuralnetwork",
                inputs=[ct.TensorType(shape=example_input.shape, name="input")]
            )
            print("✓ Converted successfully with neuralnetwork format")
            
        except Exception as e:
            print(f"✗ All conversion methods failed: {str(e)[:150]}")
            raise
    
    # Save CoreML model as .mlpackage (directory format)
    model_name = Path(pt_path).stem
    output_path = os.path.join(output_dir, f"{model_name}.mlpackage")
    
    print(f"\nSaving CoreML model as .mlpackage to: {output_path}")
    ml_model.save(output_path)
    print(f"✓ CoreML model saved successfully")
    print(f"✓ Output: {output_path}")
    
    return output_path


def main():
    """Main entry point"""
    
    # Look for .pt files in pt/ directory or project root
    project_root = Path(__file__).parent.parent
    
    # Search order: pt/ directory first, then project root
    pt_files = (
        list((project_root / "pt").glob("*.pt")) +
        list((project_root / "pt").glob("*.pth")) +
        list(project_root.glob("*.pt")) +
        list(project_root.glob("*.pth"))
    )
    
    if not pt_files:
        print("Error: No .pt/.pth files found in 'pt/' directory or project root")
        sys.exit(1)
    
    # Use most recently modified file
    pt_model_path = sorted(pt_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print(f"Found PyTorch model: {pt_model_path}\n")
    
    try:
        convert_pytorch_to_coreml(str(pt_model_path))
        print("\n✓ Conversion completed successfully!")
    except Exception as e:
        print(f"\n✗ Conversion failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
