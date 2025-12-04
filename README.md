# Amazon ML Challenge 2025 - Product Price Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.35%2B-orange)](https://huggingface.co/transformers/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-green)](https://mlflow.org/)

##  Overview

This repository contains my solution for the **Amazon ML Challenge 2025**, where the objective is to predict product prices based on catalog information. The solution uses **NeoBERT** (a model pre-trained on product catalog data) with optional **CLIP** integration for multimodal learning.

### Problem Statement
Given product catalog content (text descriptions, specifications) and optionally images, predict the product's price.

### Approach
- **Text-only Model**: Fine-tuned NeoBERT with custom regression head
- **Multimodal Model**: NeoBERT + CLIP fusion with cross-attention mechanism
- **Loss Function**: Custom SMAPE (Symmetric Mean Absolute Percentage Error)
- **Target Engineering**: Log1p transformation for price stability

---

## 🎯 Results

| Model | Validation SMAPE | Notes |
|-------|------------------|-------|
| NeoBERT (Text-only) | ~58.67% | 20 epochs, 2e-5 LR |
| NeoBERT + CLIP (Multimodal) | TBD | 12 epochs, 1e-5 LR |

---

##  Architecture

### Text-only Model (NeoBERT.py)
```
Input Text → NeoBERT Encoder → CLS Token → MLP Regressor → Price (log scale)
```

### Multimodal Model (Neobert_CLIP_fusion.py)
```
Input Text → NeoBERT → Text Features ────┐
                                          ├→ Cross Attention → MLP → Price
Input Image → CLIP Vision → Image Features ┘
```

**Key Features:**
- Cross-attention fusion mechanism
- Image augmentations (RandomResizedCrop, ColorJitter, RandomErasing)
- Optional MixUp for regularization
- Mixed precision (FP16) training
- Gradient clipping and warmup scheduling

---

## Project Structure

```
amazon-ml-challenge-2025/
│
├── dataset/
│   ├── train.csv              # Training data
│   ├── test.csv               # Test data
│   ├── train_images/          # Training images (optional)
│   └── test_images/           # Test images (optional)
│
├── src/
│   ├── NeoBERT.py            # Text-only training script
│   ├── Neobert_CLIP_fusion.py # Multimodal training script
│   ├── inference.py          # Inference script
│   └── utils.py              # Helper functions
│

├── notebooks/
│   └── Amazon_ML_2025.ipynb  # Experimentation notebook
│

├── requirements.txt
├── README.md

```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/amazon-ml-challenge-2025.git
cd amazon-ml-challenge-2025

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Place your data files in the `dataset/` folder:
- `train.csv` - Training data with columns: `sample_id`, `catalog_content`, `image_link`, `price`
- `test.csv` - Test data with columns: `sample_id`, `catalog_content`, `image_link`
- `train_images/` - (Optional) Training product images
- `test_images/` - (Optional) Test product images

### 3. Training

#### Text-only Model
```bash
python src/NeoBERT.py
```

#### Multimodal Model
```bash
python src/Neobert_CLIP_fusion.py
```

### 4. Inference

```bash
python src/inference.py \
    --model_path models/neobert/best_model.pth \
    --test_csv dataset/test.csv \
    --output_csv submission/submission.csv
```

---

## ⚙️ Configuration

### Text-only Model Parameters

```python
params = {
    'MODEL_NAME': 'chandar-lab/NeoBERT',
    'BATCH_SIZE': 64,
    'LEARNING_RATE': 2e-5,
    'NUM_EPOCHS': 20,
    'MAX_LENGTH': 625,
    'regressor_width': 256,
    'regressor_depth': 1,
    'WEIGHT_DECAY': 0.03,
    'DROPOUT_RATE': 0.4,
}
```

### Multimodal Model Parameters

```python
params = {
    'TEXT_MODEL_NAME': 'chandar-lab/NeoBERT',
    'CLIP_MODEL_NAME': 'openai/clip-vit-base-patch32',
    'BATCH_SIZE': 64,
    'LEARNING_RATE': 1e-5,
    'NUM_EPOCHS': 12,
    'regressor_width': 512,
    'regressor_depth': 2,
    'DROPOUT_RATE': 0.4,
    'WEIGHT_DECAY': 0.05,
    'USE_MIXUP': False,
    'RANDOM_ERASE_PROB': 0.2,
}
```

---

## Features

### Data Augmentation (Multimodal)
- RandomResizedCrop (scale 0.7-1.0)
- RandomHorizontalFlip
- ColorJitter (brightness, contrast, saturation, hue)
- RandomErasing (probability 0.2)
- Optional MixUp

### Training Optimizations
- ✅ Mixed Precision (FP16) training
- ✅ Gradient clipping (max norm 1.0)
- ✅ Linear warmup scheduling
- ✅ Weight decay for regularization
- ✅ Early stopping (patience 4)
- ✅ MLflow experiment tracking

### Loss Function
Custom SMAPE loss operating on log-transformed prices:
```python
SMAPE = mean(|pred - true| / ((|pred| + |true|) / 2))
```

---

##  Experiment Tracking

This project uses MLflow for experiment tracking. To view results:

```bash
mlflow ui
```

Then navigate to `http://localhost:5000` in your browser.

---

## 🔧 Hardware Requirements

**Minimum:**
- GPU: NVIDIA T4 or better
- RAM: 16GB
- VRAM: 8GB

**Recommended:**
- GPU: NVIDIA A100 / H100
- RAM: 32GB
- VRAM: 16GB+

---

##  Dataset Schema

### Training Data (`train.csv`)
| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | int | Unique sample identifier |
| `catalog_content` | str | Product description and specifications |
| `image_link` | str | URL to product image |
| `price` | float | Product price (target variable) |

### Test Data (`test.csv`)
| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | int | Unique sample identifier |
| `catalog_content` | str | Product description and specifications |
| `image_link` | str | URL to product image |

---

##  Key Components

### NeoBERT Model
- **Source**: `chandar-lab/NeoBERT`
- **Pre-training**: Product catalog data
- **Hidden Size**: 768
- **Max Length**: 625 tokens (text-only), 512 tokens (multimodal)

### CLIP Model
- **Source**: `openai/clip-vit-base-patch32`
- **Vision Encoder**: ViT-B/32
- **Image Size**: 224x224
- **Status**: Frozen (by default)

### Regressor Head
- **Text-only**: 1-layer MLP (768 → 256 → 1)
- **Multimodal**: 2-layer MLP (768 → 512 → 512 → 1)
- **Activation**: ReLU
- **Dropout**: 0.4

---

##  Dependencies

Core libraries:
- `torch>=2.0.0` - Deep learning framework
- `transformers>=4.35.0` - Hugging Face models
- `accelerate>=0.24.0` - Training acceleration
- `mlflow>=2.8.0` - Experiment tracking
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical operations
- `Pillow>=10.0.0` - Image processing
- `scikit-learn>=1.3.0` - Data splitting
- `tqdm>=4.65.0` - Progress bars

See `requirements.txt` for complete list.

---

## 🎓 Methodology

### 1. Data Preprocessing
- Clean catalog content (remove extra whitespace)
- Log1p transformation on prices
- Handle missing images with black placeholders
- Tokenization with truncation and padding

### 2. Model Training
- AdamW optimizer with linear warmup
- Gradient clipping for stability
- Early stopping based on validation SMAPE
- Model checkpointing (best and last)

### 3. Inference
- Load best model checkpoint
- Batch prediction on test set
- Expm1 transformation to get final prices
- Clip negative predictions to small positive value

### 4. Evaluation
- Primary metric: SMAPE (Symmetric Mean Absolute Percentage Error)
- Secondary metrics: MAE, MSE, RMSE, R²

---

## 🔍 Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size
params['BATCH_SIZE'] = 32  # or 16

# Reduce max sequence length
params['MAX_LENGTH'] = 512  # or 256
```

### Poor Validation Performance
```python
# Increase dropout
params['DROPOUT_RATE'] = 0.5

# Increase weight decay
params['WEIGHT_DECAY'] = 0.05

# Enable data augmentation (multimodal)
params['USE_MIXUP'] = True
params['RANDOM_ERASE_PROB'] = 0.3
```

### Slow Training
```python
# Enable mixed precision
params['USE_FP16'] = True

# Increase num_workers
params['NUM_WORKERS'] = 8  # adjust based on CPU cores

# Enable persistent workers
# (already enabled in DataLoader)
```

---

## 📚 References

1. **NeoBERT**: Chandar Lab's product-focused BERT model
2. **CLIP**: OpenAI's vision-language model
3. **SMAPE**: Symmetric Mean Absolute Percentage Error metric
4. **MixUp**: Data augmentation technique for regularization

---

##  Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request


## Author

**Divyam Sharma**
- GitHub: [@sharmadivyam](https://github.com/YOUR_USERNAME)
- LinkedIn: [LinkedIn]([https://linkedin.com/in/YOUR_PROFILE](https://www.linkedin.com/in/divyam-sharma-4a8a562b0/))

---

## Acknowledgments

- Amazon ML Challenge organizers
- Hugging Face for transformers library
- OpenAI for CLIP model
- Chandar Lab for NeoBERT model



For questions or issues, please open an issue on GitHub or reach out via [your-email@example.com](mailto:your-email@example.com).

---

**Happy Predicting! 🚀**
