# Experiments Directory

This directory contains scripts and utilities for running experiments with multimodal large language models (LLMs).

## Structure

- `models/` - Scripts and files related to downloading and managing model files.
- `utils/` - Utility scripts, such as GPU detection.
- `requirements.txt` - Python dependencies required for running experiments.
- `venv/` - (Optional) Python virtual environment for dependency isolation.

## Usage

### 1. Setup Virtual Environment
```
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 2. Install Requirements
```
pip install -r requirements.txt
```

### 3. Download Models
```
python models/download_models.py --all
# Or download specific models:
python models/download_models.py --model qwen2-vl --variation 7b
```

### 4. Detect GPU
```
python utils/gpu_detector.py
```

## Notes
- Ensure you have the correct NVIDIA drivers and CUDA-enabled PyTorch for GPU support.
- Update `requirements.txt` if you add new dependencies. 