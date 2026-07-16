"""
Training loop for Earnings Call Anomaly Detector
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import mlflow
import os
import json
import sys
sys.path.insert(0, ".")
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


class EarningsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.inputs = [tokenizer.encode(t) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.inputs[idx]),
            torch.FloatTensor([self.labels[idx]])
        )


def train():
    from model.detector import EarningsAnomalyDetector, FinancialTokenizer

    CONFIG = {
        "vocab_size": 4000,
        "d_model": 64,
        "num_heads": 4,
        "num_layers": 3,
        "d_ff": 256,
        "max_len": 128,
        "dropout": 0.2,
        "batch_size": 32,
        "lr": 2e-4,
        "epochs": 30,
        "patience": 6,
    }

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    df = pd.read_csv("data/processed/hard_synthetic.csv")
    print(f"Dataset: {len(df)} samples, fraud rate: {df['fraud_label'].mean():.1%}")

    texts = df["text"].tolist()
    labels = df["fraud_label"].tolist()

    # Train tokenizer
    tokenizer = FinancialTokenizer(
        vocab_size=CONFIG["vocab_size"],
        max_len=CONFIG["max_len"]
    )

    # Also add FIQA text if available
    try:
        fiqa = pd.read_csv("data/processed/fiqa_features.csv")
        if "text" in fiqa.columns:
            all_texts = texts + fiqa["text"].dropna().tolist()
        else:
            all_texts = texts
    except:
        all_texts = texts

    tokenizer.train(all_texts)
    os.makedirs("model", exist_ok=True)
    tokenizer.save("model/tokenizer.json")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=0.5, random_state=42
    )

    train_loader = DataLoader(
        EarningsDataset(X_train, y_train, tokenizer),
        batch_size=CONFIG["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        EarningsDataset(X_val, y_val, tokenizer),
        batch_size=CONFIG["batch_size"]
    )
    test_loader = DataLoader(
        EarningsDataset(X_test, y_test, tokenizer),
        batch_size=CONFIG["batch_size"]
    )

    model = EarningsAnomalyDetector(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        d_ff=CONFIG["d_ff"],
        max_len=CONFIG["max_len"],
        dropout=CONFIG["dropout"],
    ).to(device)

    print(f"Parameters: {model.count_parameters():,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)

    mlflow.set_experiment("earnings-anomaly-detector")

    with mlflow.start_run():
        mlflow.log_params(CONFIG)

        best_val_loss = float("inf")
        patience_counter = 0
        os.makedirs("model/checkpoints", exist_ok=True)

        for epoch in range(1, CONFIG["epochs"] + 1):
            model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                out = model(X_batch)
                loss = criterion(out["fraud_logit"], y_batch.squeeze())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            val_losses, val_preds, val_true = [], [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    out = model(X_batch)
                    loss = criterion(out["fraud_logit"], y_batch.squeeze())
                    val_losses.append(loss.item())
                    val_preds.extend(out["fraud_prob"].cpu().numpy())
                    val_true.extend(y_batch.cpu().numpy())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            val_auc = roc_auc_score(val_true, val_preds)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
            }, step=epoch)

            print(f"Epoch {epoch:3d}/{CONFIG['epochs']} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | AUC: {val_auc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "model/checkpoints/best_model.pt")
                print(f"  New best saved (AUC={val_auc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= CONFIG["patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Test evaluation
        print("\nTest evaluation...")
        model.load_state_dict(torch.load("model/checkpoints/best_model.pt", map_location=device))
        model.eval()

        test_preds, test_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                out = model(X_batch)
                test_preds.extend(out["fraud_prob"].cpu().numpy())
                test_true.extend(y_batch.numpy())

        test_auc = roc_auc_score(test_true, test_preds)
        test_binary = [1 if p > 0.5 else 0 for p in test_preds]

        print(f"\nTest AUC: {test_auc:.4f}")
        print(classification_report(test_true, test_binary, target_names=["clean", "fraud"]))

        results = {
            "test_auc": test_auc,
            "best_val_loss": best_val_loss,
            "model_parameters": model.count_parameters(),
            "epochs_trained": epoch,
        }
        with open("model/results.json", "w") as f:
            json.dump(results, f, indent=2)

        mlflow.log_metrics({"test_auc": test_auc})
        print(f"\nTraining complete! Test AUC: {test_auc:.4f}")


if __name__ == "__main__":
    train()