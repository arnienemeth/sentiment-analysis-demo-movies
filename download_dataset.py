# download_dataset.py
"""
Download Rotten Tomatoes sentiment dataset from Hugging Face.
Dataset: cornell-movie-review-data/rotten_tomatoes
"""

from datasets import load_dataset
import pandas as pd
import os


def download_and_save_dataset(output_dir: str = "data") -> dict:
    """
    Download Rotten Tomatoes dataset and save as CSV files.
    
    Args:
        output_dir: Directory to save CSV files
        
    Returns:
        Dictionary with dataset statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download dataset
    print("Downloading Rotten Tomatoes dataset...")
    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
    
    # Display dataset structure
    print(f"\nDataset structure:")
    print(dataset)
    
    # Show sample
    print(f"\nSample review:")
    print(f"  Text: {dataset['train'][0]['text']}")
    print(f"  Label: {dataset['train'][0]['label']} (0=negative, 1=positive)")
    
    # Convert to DataFrames
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    val_df = pd.DataFrame(dataset['validation'])
    
    # Save to CSV
    train_path = os.path.join(output_dir, "rt_train.csv")
    test_path = os.path.join(output_dir, "rt_test.csv")
    val_path = os.path.join(output_dir, "rt_validation.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    # Statistics
    stats = {
        "train": len(train_df),
        "test": len(test_df),
        "validation": len(val_df),
        "total": len(train_df) + len(test_df) + len(val_df)
    }
    
    print(f"\n✅ Dataset saved successfully:")
    print(f"   - {train_path} ({stats['train']} rows)")
    print(f"   - {test_path} ({stats['test']} rows)")
    print(f"   - {val_path} ({stats['validation']} rows)")
    print(f"   - Total: {stats['total']} reviews")
    
    return stats


if __name__ == "__main__":
    download_and_save_dataset()