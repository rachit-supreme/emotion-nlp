# Emotion Analysis using BERT (NLP)

A deep learning project that performs emotion classification on text using BERT (Bidirectional Encoder Representations from Transformers). The model can classify text into six different emotions: anger, fear, joy, love, sadness, and surprise.

## Overview

The project uses a fine-tuned BERT model to detect emotions in text. It achieves this by:
- Fine-tuning the BERT base model for sequence classification
- Supporting 6 emotion categories
- Using PyTorch for model training and inference
- Providing both training and inference capabilities

## Setup

1. Install required dependencies:
```sh
pip install transformers torch pandas numpy matplotlib seaborn scikit-learn tqdm
```

2. Directory structure:
```
emotion-nlp/
├── dataset/
│   ├── train.txt
│   ├── test.txt
│   └── val.txt
├── mood_analysis.py
└── README.md
```

## Usage

### Training

The model can be trained using:

```python
python mood_analysis.py
```

Training parameters:
- Batch size: 32
- Learning rate: 2e-5
- Epochs: 3
- Maximum sequence length: 256

### Inference

To analyze emotion in text:

```python
text = "Your text here"
predicted_class, probabilities = predict_emotion(text, model, tokenizer)
```

## Model Details

- Base model: BERT (bert-base-uncased)
- Fine-tuned for sequence classification
- Input: Text sequences (max length 256 tokens)
- Output: 6 emotion classes
- Model format: Both PyTorch (.safetensors) and ONNX formats available

## Visualizations

The project includes visualizations for:
- Training loss over epochs
- Learning rate changes
- Confusion matrix
- Classification reports

## Export

The model can be exported to ONNX format for deployment in different environments.
