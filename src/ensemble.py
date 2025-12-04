"""
Ensemble predictions from multiple models
Amazon ML Challenge 2025
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def load_predictions(file_paths):
    """Load prediction files"""
    predictions = []
    for path in file_paths:
        df = pd.read_csv(path)
        predictions.append(df)
    return predictions

def weighted_ensemble(predictions, weights=None):
    """
    Create weighted ensemble of predictions
    
    Args:
        predictions: List of DataFrames with 'sample_id' and 'price' columns
        weights: List of weights (if None, uses equal weights)
    
    Returns:
        DataFrame with ensemble predictions
    """
    if weights is None:
        weights = [1.0 / len(predictions)] * len(predictions)
    
    assert len(predictions) == len(weights), "Number of predictions must match weights"
    assert abs(sum(weights) - 1.0) < 1e-6, f"Weights must sum to 1, got {sum(weights)}"
    
    # Verify all predictions have same sample_ids
    sample_ids = predictions[0]['sample_id'].values
    for i, pred in enumerate(predictions[1:], 1):
        assert np.array_equal(pred['sample_id'].values, sample_ids), \
            f"Sample IDs don't match between prediction 0 and {i}"
    
    # Weighted average
    ensemble_prices = np.zeros(len(sample_ids))
    for pred, weight in zip(predictions, weights):
        ensemble_prices += weight * pred['price'].values
    
    return pd.DataFrame({
        'sample_id': sample_ids,
        'price': ensemble_prices
    })

def geometric_mean_ensemble(predictions):
    """Create geometric mean ensemble"""
    sample_ids = predictions[0]['sample_id'].values
    
    # Stack all predictions
    all_prices = np.stack([pred['price'].values for pred in predictions])
    
    # Geometric mean
    ensemble_prices = np.exp(np.mean(np.log(all_prices + 1e-8), axis=0))
    
    return pd.DataFrame({
        'sample_id': sample_ids,
        'price': ensemble_prices
    })

def median_ensemble(predictions):
    """Create median ensemble"""
    sample_ids = predictions[0]['sample_id'].values
    
    # Stack all predictions
    all_prices = np.stack([pred['price'].values for pred in predictions])
    
    # Median
    ensemble_prices = np.median(all_prices, axis=0)
    
    return pd.DataFrame({
        'sample_id': sample_ids,
        'price': ensemble_prices
    })

def rank_average_ensemble(predictions):
    """Create rank average ensemble"""
    sample_ids = predictions[0]['sample_id'].values
    
    # Convert each prediction to ranks
    ranks = []
    for pred in predictions:
        rank = pred['price'].rank(method='average').values
        ranks.append(rank)
    
    # Average ranks
    avg_ranks = np.mean(ranks, axis=0)
    
    # Convert back to prices using average of original predictions
    avg_prices = np.mean([pred['price'].values for pred in predictions], axis=0)
    
    # Sort by average rank
    sorted_indices = np.argsort(avg_ranks)
    sorted_prices = np.sort(avg_prices)
    
    # Map back to original order
    final_prices = np.zeros_like(avg_prices)
    for i, idx in enumerate(sorted_indices):
        final_prices[idx] = sorted_prices[i]
    
    return pd.DataFrame({
        'sample_id': sample_ids,
        'price': final_prices
    })

def main():
    parser = argparse.ArgumentParser(description='Ensemble model predictions')
    parser.add_argument('--predictions', nargs='+', required=True, 
                       help='List of prediction CSV files')
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                       help='Weights for each model (must sum to 1)')
    parser.add_argument('--method', choices=['weighted', 'geometric', 'median', 'rank'],
                       default='weighted', help='Ensemble method')
    parser.add_argument('--output', type=str, default='submission/ensemble.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading {len(args.predictions)} prediction files...")
    predictions = load_predictions(args.predictions)
    
    # Create ensemble
    print(f"Creating {args.method} ensemble...")
    if args.method == 'weighted':
        ensemble = weighted_ensemble(predictions, args.weights)
    elif args.method == 'geometric':
        ensemble = geometric_mean_ensemble(predictions)
    elif args.method == 'median':
        ensemble = median_ensemble(predictions)
    elif args.method == 'rank':
        ensemble = rank_average_ensemble(predictions)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble.to_csv(output_path, index=False)
    print(f"✅ Ensemble saved to {output_path}")
    
    # Print statistics
    print("\n📊 Ensemble Statistics:")
    print(ensemble['price'].describe())
    
    # Compare with individual predictions
    print("\n📈 Individual Model Statistics:")
    for i, (pred, path) in enumerate(zip(predictions, args.predictions)):
        print(f"\nModel {i+1} ({Path(path).name}):")
        print(pred['price'].describe())

if __name__ == "__main__":
    main()