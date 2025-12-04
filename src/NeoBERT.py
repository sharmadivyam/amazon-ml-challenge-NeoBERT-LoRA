"""
NeoBERT Fine-tuning for Amazon Product Price Prediction
Amazon ML Challenge 2025 - Text-only Model
"""

import os
import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import mlflow
import matplotlib.pyplot as plt
import numpy as np

# ================== MODEL DEFINITION ==================
class Regressor(torch.nn.Module):
    def __init__(self, input_dim, width, depth, dropout_rate):
        super(Regressor, self).__init__()
        layers = []
        layers.append(torch.nn.Linear(input_dim, width))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout_rate))
        for _ in range(depth - 1):
            layers.append(torch.nn.Linear(width, width))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate))
        layers.append(torch.nn.Linear(width, 1))
        self.regressor = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.regressor(x)

# ================== LOSS FUNCTION ==================
class SMAPELoss(torch.nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_true_real = torch.expm1(y_true)
        y_pred_real = torch.expm1(y_pred)
        numerator = torch.abs(y_pred_real - y_true_real)
        denominator = (torch.abs(y_true_real) + torch.abs(y_pred_real)) / 2
        loss = numerator / (denominator + self.epsilon)
        return torch.mean(loss)

# ================== DATASET ==================
class PriceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=625):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['catalog_content']
        price = torch.tensor(row['price'], dtype=torch.float)
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'price': price
        }

# ================== MAIN TRAINING ==================
def main(params):
    with mlflow.start_run(run_name=params['RUN_NAME']):
        mlflow.log_params(params)
        
        print("Loading and preprocessing data...")
        train_df = pd.read_csv(os.path.join(params['DATASET_FOLDER'], 'train.csv'))
        train_df['catalog_content'] = train_df['catalog_content'].astype(str)
        train_df = train_df.dropna(subset=['catalog_content', 'price'])
        train_df['price'] = np.log1p(train_df['price'])
        
        if params['DATA_PERCENTAGE'] < 1.0:
            train_df = train_df.sample(frac=params['DATA_PERCENTAGE'], random_state=42)
        
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
        
        print("Initializing model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(params['MODEL_NAME'], trust_remote_code=True)
        model = AutoModel.from_pretrained(params['MODEL_NAME'], trust_remote_code=True)
        
        regressor = Regressor(
            model.config.hidden_size, 
            params['regressor_width'], 
            params['regressor_depth'], 
            params['DROPOUT_RATE']
        )
        model.add_module("regressor", regressor)
        
        print("Creating datasets and dataloaders...")
        train_dataset = PriceDataset(train_df, tokenizer, params.get('MAX_LENGTH', 625))
        val_dataset = PriceDataset(val_df, tokenizer, params.get('MAX_LENGTH', 625))
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=params['BATCH_SIZE'],
            shuffle=True,
            num_workers=params.get('NUM_WORKERS', 16),
            pin_memory=True
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=params['BATCH_SIZE'],
            num_workers=params.get('NUM_WORKERS', 16),
            pin_memory=True
        )
        
        print("Starting fine-tuning...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        optimizer = AdamW(
            model.parameters(), 
            lr=params['LEARNING_RATE'], 
            weight_decay=params['WEIGHT_DECAY']
        )
        loss_fn = SMAPELoss()
        mse_loss_fn = torch.nn.MSELoss()
        scaler = torch.amp.GradScaler("cuda")
        
        epoch_metrics = []
        best_val_loss = float('inf')
        
        # Training Loop
        for epoch in range(params['NUM_EPOCHS']):
            model.train()
            total_loss = 0.0
            total_mse_loss = 0.0
            
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{params['NUM_EPOCHS']}"):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                prices = batch['price'].to(device)
                
                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    cls_embedding = outputs.last_hidden_state[:, 0, :]
                    predicted_price = model.regressor(cls_embedding).squeeze(-1)
                    loss = loss_fn(predicted_price, prices)
                    mse_loss = mse_loss_fn(predicted_price, prices)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                total_mse_loss += mse_loss.item()
            
            avg_train_loss = total_loss / len(train_dataloader)
            avg_train_mse = total_mse_loss / len(train_dataloader)
            
            print(f"Epoch {epoch + 1} - Avg Training SMAPE Loss: {avg_train_loss:.4f}")
            print(f"Epoch {epoch + 1} - Avg Training MSE Loss: {avg_train_mse:.4f}")
            print(f"Epoch {epoch + 1} - Gradient Norm: {grad_norm:.4f}")
            
            mlflow.log_metric("avg_train_smape_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("avg_train_mse_loss", avg_train_mse, step=epoch)
            mlflow.log_metric("grad_norm", grad_norm, step=epoch)
            
            # Validation
            model.eval()
            total_val_loss = 0.0
            all_prices = []
            all_predicted_prices = []
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    prices = batch['price'].to(device)
                    
                    with torch.amp.autocast("cuda"):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        cls_embedding = outputs.last_hidden_state[:, 0, :]
                        predicted_price = model.regressor(cls_embedding).squeeze(-1)
                        val_loss = loss_fn(predicted_price, prices)
                    
                    total_val_loss += val_loss.item()
                    all_prices.extend(prices.detach().cpu().numpy())
                    all_predicted_prices.extend(predicted_price.detach().cpu().float().numpy())
            
            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}")
            mlflow.log_metric("avg_val_loss", avg_val_loss, step=epoch)
            
            epoch_metrics.append({
                'epoch': epoch + 1, 
                'avg_train_loss': avg_train_loss, 
                'avg_val_loss': avg_val_loss
            })
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_output_dir = params.get('OUTPUT_DIR', 'models/neobert')
                os.makedirs(model_output_dir, exist_ok=True)
                model.save_pretrained(model_output_dir)
                torch.save(model.regressor.state_dict(), os.path.join(model_output_dir, 'regressor.pt'))
                print(f"✅ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save metrics
        print("Saving metrics to CSV...")
        metrics_df = pd.DataFrame(epoch_metrics)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        metrics_folder = "metrics"
        os.makedirs(metrics_folder, exist_ok=True)
        metrics_filename = os.path.join(metrics_folder, f"metrics_{params['RUN_NAME']}_{timestamp}.csv")
        metrics_df.to_csv(metrics_filename, index=False)
        mlflow.log_artifact(metrics_filename)
        
        # Plot predictions
        all_prices = np.expm1(all_prices)
        all_predicted_prices = np.expm1(all_predicted_prices)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(all_prices, all_predicted_prices, alpha=0.5)
        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.title("Actual vs. Predicted Prices")
        plot_path = "validation_plot.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()
        
        # Log model artifacts
        print("Logging model artifacts to MLflow...")
        mlflow.log_artifacts(model_output_dir, artifact_path="NeoBERT-finetuned")
        
        print("✅ Training complete!")

# ================== INFERENCE ==================
def run_inference(model_path, test_csv, output_csv, params):
    """Run inference on test set"""
    print("\n🔮 Starting inference on test set...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(params['MODEL_NAME'], trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    
    regressor_state = torch.load(os.path.join(model_path, 'regressor.pt'))
    regressor = Regressor(
        model.config.hidden_size, 
        params['regressor_width'], 
        params['regressor_depth'], 
        params['DROPOUT_RATE']
    )
    regressor.load_state_dict(regressor_state)
    model.add_module("regressor", regressor)
    model.to(device)
    model.eval()
    
    # Load test data
    test_df = pd.read_csv(test_csv)
    test_df['catalog_content'] = test_df['catalog_content'].astype(str)
    
    class TestDataset(Dataset):
        def __init__(self, dataframe, tokenizer, max_length=512):
            self.dataframe = dataframe
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.dataframe)
        
        def __getitem__(self, idx):
            row = self.dataframe.iloc[idx]
            inputs = self.tokenizer(
                row['catalog_content'], 
                return_tensors="pt", 
                padding='max_length', 
                truncation=True, 
                max_length=self.max_length
            )
            return {
                'sample_id': row['sample_id'],
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
            }
    
    test_dataset = TestDataset(test_df, tokenizer, params.get('MAX_LENGTH', 512))
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=params['BATCH_SIZE'], 
        shuffle=False, 
        num_workers=params.get('NUM_WORKERS', 16)
    )
    
    # Inference
    all_sample_ids = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Inference"):
            sample_ids = batch['sample_id']
            if isinstance(sample_ids, torch.Tensor):
                all_sample_ids.extend(sample_ids.cpu().tolist())
            else:
                all_sample_ids.extend(sample_ids)
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                predicted_price_log = model.regressor(cls_embedding).squeeze(-1)
            
            all_predictions.extend(predicted_price_log.detach().cpu().float().numpy())
    
    # Create submission
    final_predictions = np.expm1(all_predictions)
    submission_df = pd.DataFrame({
        'sample_id': all_sample_ids, 
        'price': final_predictions
    })
    
    submission_df.to_csv(output_csv, index=False)
    print(f"✅ Submission saved to {output_csv}")
    return submission_df

if __name__ == "__main__":
    params = {
        'RUN_NAME': 'NeoBERT_Text_Only',
        'DATASET_FOLDER': 'dataset/',
        'MODEL_NAME': 'chandar-lab/NeoBERT',
        'BATCH_SIZE': 64,
        'LEARNING_RATE': 2e-5,
        'NUM_EPOCHS': 20,
        'DATA_PERCENTAGE': 1.0,
        'MAX_LENGTH': 625,
        'regressor_width': 256,
        'regressor_depth': 1,
        'WEIGHT_DECAY': 0.03,
        'DROPOUT_RATE': 0.4,
        'NUM_WORKERS': 16,
        'OUTPUT_DIR': 'models/neobert'
    }
    
    mlflow.set_experiment("NeoBERT Price Prediction")
    main(params)