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


def infer_num_classes_from_config(config: dict) -> int:
    """
    Extract num_classes from training config if available.
    
    Args:
        config: Training configuration dict
        
    Returns:
        Number of classes, or None if not found
    """
    if not isinstance(config, dict):
        return None
    
    if 'num_classes' in config:
        num_classes = config['num_classes']
        print(f"✓ Found num_classes={num_classes} in checkpoint config")
        return num_classes
    
    return None


def infer_num_classes_from_state_dict(state_dict: dict) -> int:
    """
    Infer number of classes from state_dict by examining classifier weights.
    Prioritizes the FINAL classifier layer (output layer) to avoid intermediate hidden layers.
    
    Args:
        state_dict: The model's state_dict
        
    Returns:
        Number of classes (out_features of classifier layer)
        
    Raises:
        ValueError: If classifier weights cannot be found
    """
    # Strategy 1: Look for final classifier patterns (most reliable)
    final_classifier_patterns = [
        ('classifier.6.weight', 'MobileNetV3Small default'),  # Last layer
        ('head.weight', 'head layer'),
        ('fc.weight', 'fc layer'),
    ]
    
    for pattern, desc in final_classifier_patterns:
        if pattern in state_dict:
            weight_tensor = state_dict[pattern]
            num_classes = weight_tensor.shape[0]
            print(f"✓ Inferred num_classes={num_classes} from '{pattern}' ({desc})")
            return num_classes
    
    # Strategy 2: Search for classifier weights with custom backbone prefix
    # Find ALL classifier weights and pick the one with smallest output (most likely final layer)
    classifier_weights = {}
    for key in state_dict.keys():
        if 'classifier' in key and key.endswith('.weight') and 'bias' not in key:
            weight_tensor = state_dict[key]
            if weight_tensor.ndim >= 2:
                out_features = weight_tensor.shape[0]
                # Extract layer index from key if possible (e.g., 'classifier.4.weight' -> 4)
                try:
                    # Split by '.' and find numeric indices
                    parts = key.split('.')
                    indices = [int(p) for p in parts if p.isdigit()]
                    layer_idx = indices[-1] if indices else float('inf')
                except:
                    layer_idx = float('inf')
                
                classifier_weights[key] = (out_features, layer_idx)
    
    if classifier_weights:
        # Sort by: first priority = layer index descending (later layers), second = out_features ascending
        # (final output layers typically have fewer features than intermediate layers)
        sorted_weights = sorted(classifier_weights.items(), key=lambda x: (-x[1][1], x[1][0]))
        best_key, (num_classes, layer_idx) = sorted_weights[0]
        print(f"✓ Inferred num_classes={num_classes} from '{best_key}' (layer_idx={layer_idx})")
        return num_classes
    
    raise ValueError(
        "Could not infer num_classes from state_dict. "
        "Classifier weight layer not found. "
        f"Available keys: {list(state_dict.keys())[:10]}"
    )


def validate_model_output(model: nn.Module, example_input: torch.Tensor) -> None:
    """
    Validate that model output has correct shape before conversion.
    
    Args:
        model: PyTorch model
        example_input: Example input tensor (batch_size, channels, height, width)
        
    Raises:
        AssertionError: If output shape is invalid
    """
    print("Validating model output shape...")
    with torch.no_grad():
        output = model(example_input)
    
    assert output.ndim == 2, \
        f"Expected output shape (batch_size, num_classes), got ndim={output.ndim}, shape={output.shape}"
    
    assert output.shape[0] == 1, \
        f"Expected batch_size=1, got batch_size={output.shape[0]}"
    
    num_classes = output.shape[1]
    print(f"✓ Output validation passed: shape={output.shape}, num_classes={num_classes}")


def freeze_model(model: nn.Module) -> None:
    """
    Freeze model parameters for inference (eval mode + no gradients).
    
    Args:
        model: PyTorch model to freeze
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print("✓ Model frozen (eval mode + requires_grad=False)")


def log_coreml_metadata(ml_model, input_name: str, output_name: str) -> None:
    """
    Log CoreML model metadata for debugging.
    
    Args:
        ml_model: Converted CoreML model
        input_name: Input feature name
        output_name: Output feature name
    """
    print("\n" + "="*60)
    print("CoreML Model Metadata")
    print("="*60)
    
    model_type = type(ml_model).__name__
    print(f"Model type: {model_type}")
    
    try:
        # Log inputs
        if hasattr(ml_model, 'input_description'):
            print(f"\nInputs:")
            for inp in ml_model.input_description:
                if hasattr(inp, 'name') and hasattr(inp, 'type'):
                    print(f"  - {inp.name}: {inp.type}")
                else:
                    print(f"  - {inp}")
        
        # Log outputs
        if hasattr(ml_model, 'output_description'):
            print(f"\nOutputs:")
            for out in ml_model.output_description:
                if hasattr(out, 'name') and hasattr(out, 'type'):
                    print(f"  - {out.name}: {out.type}")
                else:
                    print(f"  - {out}")
        
        # Log spec details
        if hasattr(ml_model, 'spec'):
            spec = ml_model.spec
            try:
                if spec.HasField('neuralNetwork'):
                    print(f"\n⚠ WARNING: Model uses legacy neuralnetwork format")
                    print(f"  ANE optimization is NOT available")
                    print(f"  Consider updating conversion strategy for mlprogram format")
                elif spec.HasField('mlProgram'):
                    print(f"\n✓ Model uses mlprogram format (optimized)")
            except Exception as spec_e:
                print(f"⚠ Could not determine model format: {type(spec_e).__name__}")
    
    except Exception as e:
        print(f"⚠ Error logging metadata: {str(e)[:100]}")
    
    print("="*60)


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
    config = None
    num_classes = None
    
    # Extract model, state_dict, and config from checkpoint
    if isinstance(loaded, nn.Module):
        model = loaded
        print("✓ Direct nn.Module loaded")
    
    elif isinstance(loaded, dict):
        print(f"Checkpoint keys: {list(loaded.keys())}")
        
        # Extract config if available
        if 'config' in loaded:
            config = loaded['config']
            print(f"✓ Found training config")
        
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
        
        # MANDATORY: Infer num_classes (try config first, then state_dict)
        try:
            # Priority 1: Extract from config
            if config is not None:
                num_classes = infer_num_classes_from_config(config)
            
            # Priority 2: Infer from state_dict
            if num_classes is None:
                num_classes = infer_num_classes_from_state_dict(state_dict)
        except ValueError as e:
            print(f"✗ Failed to infer num_classes: {e}")
            raise
        
        try:
            from torchvision.models import mobilenet_v3_small
            
            # Check if state_dict has 'backbone.' prefix (custom wrapper)
            has_backbone_prefix = any(k.startswith('backbone.') for k in state_dict.keys())
            if has_backbone_prefix:
                print("Detected 'backbone.' prefix in state_dict, stripping...")
                # Remove 'backbone.' prefix to match MobileNetV3Small keys
                state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            
            # Create MobileNetV3Small with INFERRED num_classes
            model = mobilenet_v3_small(weights=None, num_classes=num_classes)
            print(f"✓ Created MobileNetV3Small architecture (num_classes={num_classes})")
            
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
                    print("⚠ Note: Classifier head was custom, using MobileNetV3Small classifier with inferred classes")
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
    
    # MANDATORY: Freeze model before conversion
    freeze_model(model)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # RECOMMENDED: Allow INPUT_SIZE override via environment variable
    input_size = int(os.getenv("INPUT_SIZE", "128"))
    print(f"\nUsing input size: {input_size}×{input_size} (override via INPUT_SIZE env var)")
    
    # Create example input
    print("Preparing example input for model conversion...")
    example_input = torch.randn(1, 3, input_size, input_size)
    print(f"Example input shape: {example_input.shape}")
    
    # MANDATORY: Validate output shape before conversion
    validate_model_output(model, example_input)
    
    ml_model = None
    conversion_method = None
    
    # Method 1: Direct coremltools conversion (PyTorch 2.0+ native)
    print("\nMethod 1: Direct coremltools conversion...")
    try:
        ml_model = ct.convert(
            model,
            convert_to="mlprogram",
            source="pytorch",
            inputs=[ct.TensorType(shape=example_input.shape, name="input")],
            outputs=[ct.TensorType(name="logits")]
        )
        print("✓ Converted successfully with mlprogram format")
        conversion_method = "mlprogram (direct)"
        
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
                inputs=[ct.TensorType(shape=example_input.shape, name="input")],
                outputs=[ct.TensorType(name="logits")]
            )
            print("✓ Converted successfully with torch.jit.trace + mlprogram")
            conversion_method = "mlprogram (jit.trace)"
            
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
                inputs=[ct.TensorType(shape=example_input.shape, name="input")],
                outputs=[ct.TensorType(name="logits")]
            )
            print("✓ Converted successfully with torch.jit.trace + neuralnetwork")
            conversion_method = "neuralnetwork (jit.trace)"
            
        except Exception as e:
            print(f"⚠ torch.jit.trace + neuralnetwork failed: {str(e)[:150]}")
    
    # Method 4: Direct conversion to neuralnetwork (legacy)
    if ml_model is None:
        print("\nMethod 4: Direct neuralnetwork conversion...")
        try:
            ml_model = ct.convert(
                model,
                convert_to="neuralnetwork",
                source="pytorch",
                inputs=[ct.TensorType(shape=example_input.shape, name="input")],
                outputs=[ct.TensorType(name="logits")]
            )
            print("✓ Converted successfully with neuralnetwork format")
            conversion_method = "neuralnetwork (direct)"
            
        except Exception as e:
            print(f"✗ All conversion methods failed: {str(e)[:150]}")
            raise
    
    # RECOMMENDED: Log CoreML metadata
    log_coreml_metadata(ml_model, "input", "logits")
    
    # Save CoreML model as .mlpackage (directory format)
    model_name = Path(pt_path).stem
    output_path = os.path.join(output_dir, f"{model_name}.mlpackage")
    
    print(f"\nSaving CoreML model as .mlpackage to: {output_path}")
    ml_model.save(output_path)
    print(f"✓ CoreML model saved successfully")
    print(f"✓ Output: {output_path}")
    print(f"✓ Conversion method: {conversion_method}")
    
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
