# LongDocURL Dataset Download

This directory contains scripts to download the LongDocURL dataset from Hugging Face.

## Files

- `requirements.txt` - Required Python packages
- `download_dataset.py` - Script to download the dataset
- `setup_and_download.py` - Complete setup script that installs dependencies and downloads the dataset
- `README.md` - This file

## Quick Start

### Option 1: Automatic Setup (Recommended)

Run the complete setup script:

```bash
python setup_and_download.py
```

This will:
1. Install all required dependencies
2. Download the LongDocURL dataset
3. Save it to the `LongDocURL/` directory

### Option 2: Manual Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Login to Hugging Face (if not already logged in):
```bash
huggingface-cli login
```

3. Download the dataset:
```bash
python download_dataset.py
```

## Dataset Information

The LongDocURL dataset will be downloaded from `dengchao/LongDocURL` on Hugging Face and saved locally in the `LongDocURL/` directory.

## Requirements

- Python 3.7+
- Internet connection
- Hugging Face account (for dataset access)

## Troubleshooting

- If you get authentication errors, make sure you're logged in to Hugging Face using `huggingface-cli login`
- If you get import errors, make sure all requirements are installed using `pip install -r requirements.txt` 