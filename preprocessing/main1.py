import os
import torch
from data_loader import load_dataset_in_chunks, load_processed_chunks
from feature_engineering import generate_simple_features, combine_features
from data_splitter import split_and_save_data, load_split_data

def main():
    # Configuration
    data_root = "./data"          # Directory to store the downloaded dataset
    processed_dir = "./processed"  # Directory to store processed data
    dataset_name = "Reddit"       # Name of the dataset

    # Step 1: Create directories
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Step 2: Load dataset in chunks
    print("\n=== STEP 1: Loading Dataset ===")
    data, chunk_info = load_dataset_in_chunks(data_root, dataset_name, save_dir=processed_dir)
    
    # Step 3: Generate simple features (much faster and more memory efficient)
    print("\n=== STEP 2: Generating Node Features ===")
    generate_simple_features(processed_dir)
    
    # Step 4: Combine original and topological features
    print("\n=== STEP 3: Combining Features ===")
    combine_features(processed_dir)
    
    # Step 5: Split data into train/val/test sets
    print("\n=== STEP 4: Splitting Data ===")
    train_data, val_data, test_data = split_and_save_data(
        processed_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        neg_ratio=1.0
    )
    
    print("\n=== Processing Complete! ===")
    print(f"Data has been processed and saved to {processed_dir}")
    print(f"Train data: {train_data.edge_label_index.size(1)} edges")
    print(f"Validation data: {val_data.edge_label_index.size(1)} edges")
    print(f"Test data: {test_data.edge_label_index.size(1)} edges")

if __name__ == "__main__":
    main()