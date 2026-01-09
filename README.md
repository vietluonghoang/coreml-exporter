# CoreML Exporter

Automated conversion of PyTorch models to CoreML format using GitHub Actions on macOS.

## Strategy

**Convert PyTorch (.pt) → CoreML (.mlmodel)**

This approach bypasses the ONNX IR v10 incompatibility with coremltools that was causing segmentation faults.

## Structure

```
.
├── pt/                                 # PyTorch model files
│   └── best_model.pt                   # Input model
├── scripts/
│   └── export_pytorch_to_coreml.py     # Main conversion script
├── output/                              # Generated CoreML models
├── .github/workflows/
│   └── export-to-coreml.yml            # GitHub Actions workflow
├── requirements.txt
└── README.md
```

## Local Usage

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Conversion

```bash
python scripts/export_pytorch_to_coreml.py
```

The converted CoreML model will be saved to the `output/` directory as `.mlpackage` format (directory structure).

## Conversion Methods (Fallback Chain)

The script tries 4 methods in sequence:

1. **Direct coremltools conversion** - Native PyTorch 2.0+ support
2. **torch.jit.trace + mlprogram** - Modern Neural Networks 5 format
3. **torch.jit.trace + neuralnetwork** - Traced model with legacy format
4. **Direct neuralnetwork conversion** - Final fallback

## GitHub Actions

The workflow is automatically triggered when:
- Changes are pushed to `main` or `develop` branches
- Changes are made to `pt/` directory
- Manual trigger via `workflow_dispatch`

### Outputs

- CoreML model artifacts available for download
- Release created on main branch with CoreML model

## Requirements

- Python 3.11+
- macOS (for GitHub Actions runner)
- PyTorch 2.0+
- coremltools 8.3+

See requirements.txt for full dependency list.

## Why PyTorch instead of ONNX?

ONNX IR v10 with MobileNetV3 causes segmentation faults in coremltools due to operator incompatibilities (HardSwish). Direct PyTorch conversion is more stable and reliable.
