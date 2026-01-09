# CoreML Exporter

Automated conversion of ONNX models (MobileNetV3Small) to CoreML format using GitHub Actions on macOS.

## Features

- Automatic model conversion on every commit
- GitHub Actions workflow on macOS runner
- Artifact storage and release management
- Easy local testing

## Structure

```
.
├── onnx/                          # ONNX model files
│   ├── model.onnx
│   └── model.onnx.data
├── scripts/
│   └── export_onnx_to_coreml.py   # Conversion script
├── output/                         # Generated CoreML models
├── .github/workflows/
│   └── export-to-coreml.yml       # GitHub Actions workflow
├── requirements.txt
└── .gitignore
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
python scripts/export_onnx_to_coreml.py
```

The converted CoreML model will be saved to the `output/` directory.

## GitHub Actions

The workflow is automatically triggered when:
- Changes are pushed to `main` or `develop` branches
- Changes are made to `onnx/` directory
- Manual trigger via `workflow_dispatch`

### Outputs

- CoreML model artifacts available for download
- Release created on main branch with CoreML model

## Requirements

- Python 3.11+
- macOS (for GitHub Actions runner)
- See requirements.txt for dependencies
