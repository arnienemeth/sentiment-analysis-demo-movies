# train_model.py
"""
Fine-tune DistilBERT for sentiment analysis on Rotten Tomatoes dataset.
Uses PyTorch + Hugging Face Transformers.
"""

import os
import torch
import pandas as pd
import boto3
from io import StringIO
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "bucket_name": "sentiment-analysis-demo-arnoldnemeth",
    "model_name": "distilbert-base-uncased",
    "output_dir": "./model_output",
    "max_length": 256,
    "batch_size": 16,
    "epochs": 2,
    "learning_rate": 2e-5,
    "use_s3_data": True,  # Set False to use local data
}


# ============================================================
# DATASET CLASS
# ============================================================

class SentimentDataset(Dataset):
    """Custom Dataset for sentiment analysis."""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

def load_data_from_s3(bucket_name: str, key: str) -> pd.DataFrame:
    """Load CSV file from S3 bucket."""
    print(f"   Loading s3://{bucket_name}/{key}...")
    s3_client = boto3.client("s3")
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    csv_content = response["Body"].read().decode("utf-8")
    return pd.read_csv(StringIO(csv_content))


def load_data_local(filepath: str) -> pd.DataFrame:
    """Load CSV file from local directory."""
    print(f"   Loading {filepath}...")
    return pd.read_csv(filepath)


def load_all_data(use_s3: bool, bucket_name: str) -> tuple:
    """Load train, validation, and test datasets."""
    print("\n📂 Loading data...")
    
    if use_s3:
        train_df = load_data_from_s3(bucket_name, "data/rt_train.csv")
        val_df = load_data_from_s3(bucket_name, "data/rt_validation.csv")
        test_df = load_data_from_s3(bucket_name, "data/rt_test.csv")
    else:
        train_df = load_data_local("data/rt_train.csv")
        val_df = load_data_local("data/rt_validation.csv")
        test_df = load_data_local("data/rt_test.csv")
    
    print(f"   ✅ Train: {len(train_df)} samples")
    print(f"   ✅ Validation: {len(val_df)} samples")
    print(f"   ✅ Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df


# ============================================================
# METRICS FUNCTION
# ============================================================

def compute_metrics(pred):
    """Compute accuracy, precision, recall, and F1 score."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    accuracy = accuracy_score(labels, preds)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train():
    """Main training pipeline."""
    
    print("=" * 60)
    print("🎬 SENTIMENT ANALYSIS MODEL TRAINING")
    print("=" * 60)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n💻 Using device: {device}")
    
    # Load data
    train_df, val_df, test_df = load_all_data(
        use_s3=CONFIG["use_s3_data"],
        bucket_name=CONFIG["bucket_name"]
    )
    
    # Initialize tokenizer
    print(f"\n🔤 Loading tokenizer: {CONFIG['model_name']}")
    tokenizer = DistilBertTokenizer.from_pretrained(CONFIG["model_name"])
    
    # Create datasets
    print("\n📊 Creating datasets...")
    train_dataset = SentimentDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=CONFIG["max_length"],
    )
    
    val_dataset = SentimentDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=CONFIG["max_length"],
    )
    
    test_dataset = SentimentDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=CONFIG["max_length"],
    )
    
    # Load model
    print(f"\n🤖 Loading model: {CONFIG['model_name']}")
    model = DistilBertForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=2,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs",
        logging_steps=100,
        report_to="none",  # Disable wandb
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("🚀 STARTING TRAINING...")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("📈 EVALUATING ON TEST SET...")
    print("=" * 60 + "\n")
    
    results = trainer.evaluate(test_dataset)
    
    print("\n📊 Test Results:")
    print(f"   Accuracy:  {results['eval_accuracy']:.4f}")
    print(f"   Precision: {results['eval_precision']:.4f}")
    print(f"   Recall:    {results['eval_recall']:.4f}")
    print(f"   F1 Score:  {results['eval_f1']:.4f}")
    
    # Save model
    print(f"\n💾 Saving model to {CONFIG['output_dir']}/final_model/")
    trainer.save_model(f"{CONFIG['output_dir']}/final_model")
    tokenizer.save_pretrained(f"{CONFIG['output_dir']}/final_model")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    
    return trainer, results


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    trainer, results = train()