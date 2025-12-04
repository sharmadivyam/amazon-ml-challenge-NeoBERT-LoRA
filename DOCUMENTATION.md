# Amazon ML Challenge 2025 - Technical Documentation

## Table of Contents
1. [Problem Overview](#problem-overview)
2. [Model Architecture](#model-architecture)
3. [Training Pipeline](#training-pipeline)
4. [Data Processing](#data-processing)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Production Deployment](#production-deployment)

---

## Problem Overview

### Task
Predict product prices based on:
- **Catalog content** (text): Product descriptions, specifications, features
- **Product images** (optional): Visual representation of products

### Challenges
1. **Skewed price distribution**: Prices range from $0.01 to $10,000+
2. **Variable text length**: Descriptions vary from 10 to 5000+ characters
3. **Missing data**: Some products lack images or complete descriptions
4. **Domain-specific language**: Technical specifications, brand names, measurements

### Evaluation Metric
**SMAPE (Symmetric Mean Absolute Percentage Error)**

```
SMAPE = (100 / n) * Σ |actual - predicted| / ((|actual| + |predicted|) / 2)
```

Lower is better (0% = perfect, 100% = worst)

---

## Model Architecture

### 1. Text-only Model (NeoBERT)

```
┌─────────────────┐
│  Catalog Text   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tokenization   │
│  (max_len=625)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  NeoBERT        │
│  Encoder        │
│  (768-dim)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CLS Token      │
│  Extraction     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MLP Regressor  │
│  768→256→1      │
│  (ReLU+Dropout) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Price (log1p)   │
└─────────────────┘
```

**Components:**
- **NeoBERT Encoder**: Pre-trained on product catalog data
- **CLS Token**: [CLS] token embedding (768-dim)
- **Regressor Head**: 
  - Layer 1: Linear(768, 256) + ReLU + Dropout(0.4)
  - Layer 2: Linear(256, 1)

**Parameters:**
- Total: ~110M
- Trainable: ~110M (full fine-tuning)

### 2. Multimodal Model (NeoBERT + CLIP)

```
┌─────────────────┐     ┌─────────────────┐
│  Catalog Text   │     │ Product Image   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  NeoBERT        │     │  CLIP Vision    │
│  Encoder        │     │  Encoder        │
│  (768-dim)      │     │  (512-dim)      │
└────────┬────────┘     └────────┬────────┘
         │                       │
         │                       ▼
         │              ┌─────────────────┐
         │              │  Projection     │
         │              │  512→768        │
         │              └────────┬────────┘
         │                       │
         ▼                       ▼
         ┌───────────────────────┐
         │  Cross Attention      │
         │  (8 heads)            │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Fused Features       │
         │  (768-dim)            │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  MLP Regressor        │
         │  768→512→512→1        │
         │  (ReLU+Dropout)       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Price (log1p)        │
         └───────────────────────┘
```

**Components:**
- **NeoBERT Encoder**: Text feature extraction (768-dim)
- **CLIP Vision**: Image feature extraction (512-dim, frozen)
- **Cross Attention**: 
  - Query: Text features
  - Key/Value: Projected image features
  - 8 attention heads
- **Regressor Head**:
  - Layer 1: Linear(768, 512) + ReLU + Dropout(0.4)
  - Layer 2: Linear(512, 512) + ReLU + Dropout(0.4)
  - Layer 3: Linear(512, 1)

**Parameters:**
- Total: ~230M
- Trainable: ~120M (CLIP vision frozen)

---

## Training Pipeline

### 1. Data Loading

```python
train_df = pd.read_csv('dataset/train.csv')
train_df['price'] = np.log1p(train_df['price'])  # Log transformation
```

### 2. Train/Val Split

```python
train_df, val_df = train_test_split(
    train_df, 
    test_size=0.1, 
    random_state=42
)
```

### 3. Data Augmentation (Multimodal Only)

**PIL-level (before CLIP processor):**
- RandomResizedCrop(224, scale=(0.7, 1.0))
- RandomHorizontalFlip()
- RandomRotation(15°)
- ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02)

**Tensor-level (after CLIP processor):**
- RandomErasing(p=0.2, scale=(0.02, 0.33))

**Optional MixUp:**
```python
if USE_MIXUP:
    lambda = Beta(alpha, alpha)
    mixed_images = lambda * images + (1-lambda) * shuffled_images
    mixed_prices = lambda * prices + (1-lambda) * shuffled_prices
```

### 4. Optimization

**Optimizer:** AdamW
```python
optimizer = AdamW(
    params=trainable_params,
    lr=2e-5,  # text-only
    weight_decay=0.03
)
```

**Learning Rate Schedule:**
```python
# Linear warmup + linear decay
warmup_steps = 0.03 * total_steps
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

### 5. Loss Function

**Custom SMAPE Loss:**
```python
def smape_loss(pred_log, true_log):
    pred = torch.expm1(pred_log)  # Convert from log space
    true = torch.expm1(true_log)
    
    numerator = torch.abs(pred - true)
    denominator = (torch.abs(pred) + torch.abs(true)) / 2
    
    return torch.mean(numerator / (denominator + eps))
```

### 6. Training Loop

```python
for epoch in range(num_epochs):
    # Training
    model.train()
    for batch in train_loader:
        with torch.cuda.amp.autocast():
            predictions = model(batch)
            loss = smape_loss(predictions, batch['price'])
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    
    # Validation
    model.eval()
    val_smape = evaluate(model, val_loader)
    
    # Save best model
    if val_smape < best_smape:
        save_checkpoint(model, epoch, val_smape)
```

---

## Data Processing

### 1. Text Processing

**Cleaning:**
```python
def clean_catalog_content(text):
    text = str(text)
    text = " ".join(text.split())  # Remove extra whitespace
    return text
```

**Tokenization:**
```python
inputs = tokenizer(
    text,
    truncation=True,
    padding='max_length',
    max_length=625,  # text-only: 625, multimodal: 512
    return_tensors='pt'
)
```

### 2. Image Processing

**Loading:**
```python
def load_image(image_link):
    # Try to load from local directory
    filename = os.path.basename(urllib.parse.urlparse(image_link).path)
    image_path = os.path.join(image_dir, filename)
    
    if os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
    else:
        # Create black placeholder
        img = Image.new("RGB", (224, 224), (0, 0, 0))
    
    # Pad to square (preserve aspect ratio)
    w, h = img.size
    max_side = max(w, h, 224)
    img = ImageOps.pad(img, (max_side, max_side), color=(0,0,0))
    
    return img
```

**CLIP Processing:**
```python
processed = clip_processor(
    images=images,
    return_tensors='pt'
)
pixel_values = processed['pixel_values']  # (B, 3, 224, 224)
```

### 3. Target Transformation

**Log1p transformation:**
```python
# Training
train_df['price'] = np.log1p(train_df['price'])

# Inference
predictions = np.expm1(log_predictions)
```

**Why log1p?**
- Handles price skewness
- Stabilizes variance
- Makes optimization easier
- SMAPE loss performs better

---

## Hyperparameter Tuning

### Text-only Model

| Hyperparameter | Tried Values | Best Value | Notes |
|----------------|--------------|------------|-------|
| Learning Rate | 1e-5, 2e-5, 5e-5 | 2e-5 | Higher LR caused instability |
| Batch Size | 32, 64, 128 | 64 | Trade-off between speed & memory |
| Max Length | 512, 625, 1024 | 625 | Longer context helps |
| Dropout | 0.2, 0.4, 0.5 | 0.4 | Good regularization |
| Weight Decay | 0.01, 0.03, 0.05 | 0.03 | Prevents overfitting |
| Regressor Width | 128, 256, 512 | 256 | Sufficient capacity |
| Regressor Depth | 1, 2, 3 | 1 | Simpler is better |

### Multimodal Model

| Hyperparameter | Tried Values | Best Value | Notes |
|----------------|--------------|------------|-------|
| Learning Rate | 5e-6, 1e-5, 2e-5 | 1e-5 | Lower due to image branch |
| Batch Size | 32, 64 | 64 | Limited by image memory |
| Dropout | 0.3, 0.4, 0.5 | 0.4 | Same as text-only |
| Weight Decay | 0.03, 0.05, 0.1 | 0.05 | Slightly higher |
| Regressor Width | 256, 512, 1024 | 512 | Larger for fusion |
| Regressor Depth | 1, 2, 3 | 2 | More complex patterns |
| MixUp Alpha | 0.2, 0.3, 0.5 | 0.3 | Moderate mixing |
| Random Erase | 0.1, 0.2, 0.3 | 0.2 | Good balance |

---

## Evaluation Metrics

### Primary Metric: SMAPE

**Definition:**
```
SMAPE = (1/n) * Σ |pred - true| / ((|pred| + |true|) / 2)
```

**Properties:**
- Symmetric: SMAPE(a,b) = SMAPE(b,a)
- Percentage-based: Easy to interpret
- Bounded: [0, 2] (reported as 0-200%)
- Handles different scales well

**Calculation:**
```python
def smape_np(preds, targs, eps=1e-6):
    preds = np.maximum(preds, eps)
    targs = np.maximum(targs, eps)
    return np.mean(
        np.abs(preds - targs) / ((np.abs(preds) + np.abs(targs)) / 2)
    )
```

### Secondary Metrics

**MAE (Mean Absolute Error):**
```python
MAE = np.mean(np.abs(pred - true))
```

**RMSE (Root Mean Squared Error):**
```python
RMSE = np.sqrt(np.mean((pred - true) ** 2))
```

**R² Score:**
```python
R2 = 1 - (SS_res / SS_tot)
```

---

## Production Deployment

### 1. Model Export

**PyTorch:**
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config
}, 'model.pth')
```

**ONNX (Optional):**
```python
import torch.onnx

dummy_input = {
    'input_ids': torch.randint(0, 30000, (1, 512)),
    'attention_mask': torch.ones(1, 512)
}

torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    opset_version=14
)
```

### 2. Inference Optimization

**Batch Processing:**
```python
batch_size = 64
for i in range(0, len(test_df), batch_size):
    batch = test_df[i:i+batch_size]
    predictions = model.predict(batch)
```

**FP16 Inference:**
```python
model.half()  # Convert to FP16
with torch.cuda.amp.autocast():
    predictions = model(inputs)
```

**TensorRT (Optional):**
```python
import torch_tensorrt

trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((1, 512))],
    enabled_precisions={torch.float16}
)
```

### 3. API Service

**FastAPI Example:**
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictRequest(BaseModel):
    catalog_content: str
    image_url: str = None

@app.post("/predict")
async def predict(request: PredictRequest):
    # Preprocess
    inputs = preprocess(request.catalog_content, request.image_url)
    
    # Predict
    with torch.no_grad():
        prediction = model(inputs)
    
    # Postprocess
    price = np.expm1(prediction.item())
    
    return {"predicted_price": float(price)}
```

### 4. Monitoring

**Metrics to Track:**
- Prediction latency (p50, p95, p99)
- Throughput (requests/second)
- Error rate
- Prediction distribution drift

**MLflow Tracking:**
```python
mlflow.log_metric("inference_latency_ms", latency)
mlflow.log_metric("prediction_mean", predictions.mean())
mlflow.log_metric("prediction_std", predictions.std())
```

---

## Best Practices

### 1. Data Quality
- ✅ Clean catalog content thoroughly
- ✅ Handle missing images gracefully
- ✅ Validate price ranges
- ✅ Remove outliers (optional)

### 2. Training
- ✅ Use mixed precision (FP16)
- ✅ Implement gradient clipping
- ✅ Use learning rate warmup
- ✅ Save multiple checkpoints
- ✅ Track experiments with MLflow

### 3. Validation
- ✅ Use stratified split if possible
- ✅ Cross-validate on multiple folds
- ✅ Analyze error patterns
- ✅ Check for data leakage

### 4. Production
- ✅ Version your models
- ✅ Monitor prediction distribution
- ✅ Implement fallback mechanisms
- ✅ Cache frequent predictions
- ✅ Set up alerts for anomalies

---

## Troubleshooting Guide

### Common Issues

**1. NaN Loss**
```
Cause: Unstable gradients or bad initialization
Solution: 
- Reduce learning rate
- Increase gradient clipping
- Check for inf/nan in data
```

**2. Overfitting**
```
Cause: Model memorizing training data
Solution:
- Increase dropout rate
- Increase weight decay
- Add more augmentation
- Reduce model capacity
```

**3. Slow Training**
```
Cause: I/O bottleneck or inefficient data loading
Solution:
- Increase num_workers
- Use persistent_workers=True
- Enable pin_memory=True
- Preprocess and cache data
```

**4. Poor Generalization**
```
Cause: Train/val distribution mismatch
Solution:
- Verify data split strategy
- Check for data leakage
- Analyze error by price range
- Use more diverse augmentation
```

---

## Future Improvements

1. **Model Architecture:**
   - Try other backbones (DeBERTa, RexBERT)
   - Experiment with attention mechanisms
   - Add product category embeddings

2. **Data:**
   - Collect more training samples
   - Improve image quality
   - Extract structured features

3. **Training:**
   - Implement curriculum learning
   - Use pseudo-labeling
   - Try knowledge distillation

4. **Ensemble:**
   - Combine multiple model types
   - Optimize ensemble weights
   - Use stacking

---

## References

1. NeoBERT: https://huggingface.co/chandar-lab/NeoBERT
2. CLIP: https://github.com/openai/CLIP
3. Transformers: https://huggingface.co/docs/transformers
4. PyTorch: https://pytorch.org/docs/
5. MLflow: https://mlflow.org/docs/

---

**Last Updated:** December 2024