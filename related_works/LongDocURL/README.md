# LongDocURL Dataset Implementation

This directory contains our implementation and evaluation of methods using the **LongDocURL** dataset for multimodal document understanding research.

## 📋 Dataset Information

**LongDocURL** is a comprehensive dataset for long document understanding that combines visual and textual information. It contains diverse document types with rich multimodal annotations.

### Source
- **Original Repository**: [https://github.com/dengc2023/LongDocURL](https://github.com/dengc2023/LongDocURL)
- **Paper**: "LongDocURL: A Long Document Understanding Dataset with Rich Layout Information"
- **Authors**: Deng et al.

### Dataset Characteristics
- **Document Types**: Academic papers, reports, forms, technical documents
- **Modalities**: Text, layout, images
- **Annotations**: Document structure, layout elements, text content
- **Size**: Large-scale dataset for comprehensive evaluation

## 📁 Directory Structure

```
related_works/LongDocURL/
├── README.md                    # This file
├── requirements.txt             # Dependencies for LongDocURL experiments
├── config/                      # Configuration files
├── data/                        # Data processing scripts
├── eval/                        # Evaluation scripts
├── scripts/                     # Utility scripts
├── utils/                       # Helper utilities
└── evaluation_results/          # Experimental results
```

## 🚀 Quick Start

### Setup
```bash
cd related_works/LongDocURL
pip install -r requirements.txt
```

### Data Processing
```bash
python data/process_dataset.py
```

### Evaluation
```bash
python eval/run_evaluation.py
```

## 🔬 Research Context

This implementation is part of our comprehensive evaluation of multimodal document processing methods. We use LongDocURL to:

- **Benchmark existing methods** against a standardized dataset
- **Compare performance** across different approaches
- **Validate our proposed method** against strong baselines
- **Analyze strengths and limitations** of current approaches

## 📊 Evaluation Metrics

We evaluate methods on LongDocURL using:
- **Document Understanding**: Accuracy, F1-score
- **Layout Recognition**: IoU, Precision, Recall
- **Information Extraction**: Entity recognition, Relation extraction
- **Computational Efficiency**: Training time, Inference speed

## 📚 References

- **Original Dataset**: [LongDocURL GitHub Repository](https://github.com/dengc2023/LongDocURL)
- **Paper**: Deng et al. "LongDocURL: A Long Document Understanding Dataset with Rich Layout Information"

## 🤝 Acknowledgments

We thank the authors of LongDocURL for providing this comprehensive dataset that enables systematic evaluation of multimodal document understanding methods.

---

**Note**: This implementation is part of our research on multimodal document processing. Please refer to the original LongDocURL repository for the official dataset and implementation. 