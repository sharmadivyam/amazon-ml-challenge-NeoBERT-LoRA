

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from NeoBERT import NeoBertRegressor, PricingDataset, collate_fn, Config

def generate_predictions(
    model_path: str,
    test_csv: str,
    output_csv: str,
    batch_size: int = 64
):
    """Generate predictions for test data"""
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {DEVICE}")
    
    # Load tokenizer and model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        Config.SAVE_DIR, 
        use_fast=True, 
        trust_remote_code=True
    )
    
    model = NeoBertRegressor(
        Config.MODEL_NAME,
        Config.DROPOUT,
        Config.HIDDEN_SIZE
    ).to(DEVICE)
    
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )
    model.eval()
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(test_csv)
    print(f"Test samples: {len(test_df)}")
    
    # Create dataset and loader
    test_ds = PricingDataset(
        test_df, 
        tokenizer, 
        Config.MAX_LEN, 
        is_train=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Generate predictions
    print("Generating predictions...")
    preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            
            with torch.cuda.amp.autocast():
                out = model(ids, mask)
            
            preds.extend(torch.expm1(out).cpu().numpy().tolist())
    
    # Post-process predictions (clip negative values)
    preds = np.maximum(np.array(preds), 1e-3)
    
    # Create submission
    submission = pd.DataFrame({
        "sample_id": test_df['sample_id'],
        "price": preds
    })
    
    submission.to_csv(output_csv, index=False)
    print(f"✅ Predictions saved to: {output_csv}")
    print(f"   Sample predictions:\n{submission.head(10)}")
    
    return submission

def main():
    # Paths
    MODEL_PATH = f"{Config.SAVE_DIR}/best_model.pth"
    TEST_CSV = f"{Config.DATA_DIR}/test.csv"
    OUTPUT_CSV = "./submission.csv"
    
    # Generate predictions
    submission = generate_predictions(
        model_path=MODEL_PATH,
        test_csv=TEST_CSV,
        output_csv=OUTPUT_CSV,
        batch_size=Config.BATCH_SIZE
    )
    
    print("\n📊 Prediction Statistics:")
    print(submission['price'].describe())

if __name__ == "__main__":
    main()