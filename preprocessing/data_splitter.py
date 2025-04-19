import torch
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data
import os

def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, neg_ratio=1.0):
    """
    Split the data into train, validation, and test sets
    
    Args:
        data (torch_geometric.data.Data): Data object to split
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        neg_ratio (float): Ratio of negative samples to positive samples
        
    Returns:
        train_data, val_data, test_data: Split data objects
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1"
    
    transform = RandomLinkSplit(
        num_val=val_ratio / (train_ratio + val_ratio),  # Adjusted for the remaining data after test split
        num_test=test_ratio,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=neg_ratio
    )
    
    print("Splitting data into train/val/test sets...")
    train_data, val_data, test_data = transform(data)
    
    # Print split statistics
    total_edges = (train_data.edge_label_index.size(1) + 
                  val_data.edge_label_index.size(1) + 
                  test_data.edge_label_index.size(1))
    
    print(f"Train edges: {train_data.edge_label_index.size(1)} ({train_data.edge_label_index.size(1)/total_edges:.2%})")
    print(f"Validation edges: {val_data.edge_label_index.size(1)} ({val_data.edge_label_index.size(1)/total_edges:.2%})")
    print(f"Test edges: {test_data.edge_label_index.size(1)} ({test_data.edge_label_index.size(1)/total_edges:.2%})")
    
    return train_data, val_data, test_data

def split_and_save_data(processed_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, neg_ratio=1.0):
    """
    Load combined features, create a Data object, split it, and save the splits
    
    Args:
        processed_dir (str): Directory with processed data
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        neg_ratio (float): Ratio of negative samples to positive samples
        
    Returns:
        None (saves split data to disk)
    """
    # Load chunk info and edge_index
    chunk_info = torch.load(f"{processed_dir}/chunk_info.pt")
    edge_index = torch.load(f"{processed_dir}/edge_index.pt")
    
    # Load combined features
    num_chunks = chunk_info['num_chunks']
    all_features = []
    
    for i in range(num_chunks):
        features_chunk = torch.load(f"{processed_dir}/combined_features_chunk_{i}.pt")
        all_features.append(features_chunk)
    
    # Concatenate all features
    x = torch.cat(all_features, dim=0)
    
    # Create a Data object
    data = Data(x=x, edge_index=edge_index)
    
    # Split the data
    train_data, val_data, test_data = split_data(
        data, train_ratio, val_ratio, test_ratio, neg_ratio
    )
    
    # Save the split data
    torch.save(train_data, f"{processed_dir}/train_data.pt")
    torch.save(val_data, f"{processed_dir}/val_data.pt")
    torch.save(test_data, f"{processed_dir}/test_data.pt")
    
    print(f"Data split and saved to {processed_dir}")
    
    return train_data, val_data, test_data

def load_split_data(processed_dir):
    """
    Load the split data from disk
    
    Args:
        processed_dir (str): Directory with processed data
        
    Returns:
        train_data, val_data, test_data: Split data objects
    """
    train_data = torch.load(f"{processed_dir}/train_data.pt")
    val_data = torch.load(f"{processed_dir}/val_data.pt")
    test_data = torch.load(f"{processed_dir}/test_data.pt")
    
    print("Loaded split data:")
    print(f"Train edges: {train_data.edge_label_index.size(1)}")
    print(f"Validation edges: {val_data.edge_label_index.size(1)}")
    print(f"Test edges: {test_data.edge_label_index.size(1)}")
    
    return train_data, val_data, test_data