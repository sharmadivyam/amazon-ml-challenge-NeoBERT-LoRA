"""
Utility Functions for Amazon ML Challenge
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

# ================== DATA CLEANING ==================
def clean_catalog_content(text: str) -> str:
    """Clean catalog content text"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    # Remove extra whitespace
    text = " ".join(text.split())
    return text

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset"""
    df = df.copy()
    
    # Clean catalog content
    df['catalog_content'] = df['catalog_content'].apply(clean_catalog_content)
    
    # Remove rows with missing critical values
    df = df.dropna(subset=['catalog_content'])
    
    if 'price' in df.columns:
        df = df.dropna(subset=['price'])
        # Remove negative or zero prices
        df = df[df['price'] > 0]
    
    return df.reset_index(drop=True)

# ================== METRICS ==================
def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE)
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        eps: Small value to avoid division by zero
    
    Returns:
        SMAPE score (0 to 1, lower is better)
    """
    y_true = np.maximum(y_true, eps)
    y_pred = np.maximum(y_pred, eps)
    
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_pred) + np.abs(y_true)) / 2
    
    return np.mean(numerator / denominator)

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error"""
    return np.sqrt(calculate_mse(y_true, y_pred))

def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² Score"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def evaluate_predictions(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate all metrics"""
    return {
        "SMAPE": calculate_smape(y_true, y_pred),
        "MAE": calculate_mae(y_true, y_pred),
        "MSE": calculate_mse(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "R2": calculate_r2(y_true, y_pred)
    }

# ================== VISUALIZATION ==================
def plot_price_distribution(df: pd.DataFrame, title: str = "Price Distribution"):
    """Plot price distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Original scale
    axes[0].hist(df['price'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Price')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{title} (Original Scale)')
    axes[0].grid(alpha=0.3)
    
    # Log scale
    axes[1].hist(np.log1p(df['price']), bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('log1p(Price)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{title} (Log Scale)')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    title: str = "Predictions vs Actual"
):
    """Plot predictions vs actual values"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=10)
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Price')
    axes[0].set_ylabel('Predicted Price')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Residual plot
    residuals = y_pred - y_true
    axes[1].scatter(y_true, residuals, alpha=0.3, s=10)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Actual Price')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray):
    """Plot error distribution"""
    errors = y_pred - y_true
    percentage_errors = 100 * errors / y_true
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute error
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Prediction Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution')
    axes[0].grid(alpha=0.3)
    
    # Percentage error
    axes[1].hist(percentage_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Percentage Error (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Percentage Error Distribution')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_training_history(history: Dict[str, List[float]]):
    """Plot training history"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # SMAPE
    if 'val_smape' in history:
        axes[1].plot(epochs, history['val_smape'], 'g-', label='Val SMAPE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('SMAPE')
        axes[1].set_title('Validation SMAPE')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ================== DATA ANALYSIS ==================
def analyze_dataset(df: pd.DataFrame) -> Dict:
    """Analyze dataset statistics"""
    stats = {
        "total_samples": len(df),
        "missing_values": df.isnull().sum().to_dict()
    }
    
    if 'price' in df.columns:
        stats["price_stats"] = {
            "mean": df['price'].mean(),
            "median": df['price'].median(),
            "std": df['price'].std(),
            "min": df['price'].min(),
            "max": df['price'].max(),
            "q1": df['price'].quantile(0.25),
            "q3": df['price'].quantile(0.75)
        }
    
    if 'catalog_content' in df.columns:
        df['text_length'] = df['catalog_content'].astype(str).str.len()
        stats["text_length_stats"] = {
            "mean": df['text_length'].mean(),
            "median": df['text_length'].median(),
            "min": df['text_length'].min(),
            "max": df['text_length'].max()
        }
    
    return stats

def print_dataset_info(df: pd.DataFrame):
    """Print dataset information"""
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    
    stats = analyze_dataset(df)
    
    print(f"\n📊 Total Samples: {stats['total_samples']}")
    
    print("\n🔍 Missing Values:")
    for col, count in stats['missing_values'].items():
        if count > 0:
            print(f"   {col}: {count} ({100*count/stats['total_samples']:.2f}%)")
    
    if 'price_stats' in stats:
        print("\n💰 Price Statistics:")
        for key, value in stats['price_stats'].items():
            print(f"   {key}: ${value:.2f}")
    
    if 'text_length_stats' in stats:
        print("\n📝 Text Length Statistics:")
        for key, value in stats['text_length_stats'].items():
            print(f"   {key}: {value:.0f} characters")
    
    print("=" * 60)

# ================== ENSEMBLE ==================
def weighted_ensemble(
    predictions: List[np.ndarray],
    weights: List[float]
) -> np.ndarray:
    """
    Create weighted ensemble of predictions
    
    Args:
        predictions: List of prediction arrays
        weights: List of weights (should sum to 1)
    
    Returns:
        Weighted average predictions
    """
    assert len(predictions) == len(weights), "Number of predictions must match weights"
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
    
    ensemble = np.zeros_like(predictions[0])
    for pred, weight in zip(predictions, weights):
        ensemble += weight * pred
    
    return ensemble

def optimize_ensemble_weights(
    predictions: List[np.ndarray],
    y_true: np.ndarray,
    metric_fn=calculate_smape
) -> List[float]:
    """
    Find optimal ensemble weights using grid search
    
    Args:
        predictions: List of prediction arrays
        y_true: Ground truth values
        metric_fn: Metric function to optimize
    
    Returns:
        Optimal weights
    """
    from scipy.optimize import minimize
    
    def objective(weights):
        ensemble = weighted_ensemble(predictions, weights)
        return metric_fn(y_true, ensemble)
    
    n_models = len(predictions)
    initial_weights = [1.0 / n_models] * n_models
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_models)]
    
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x.tolist()