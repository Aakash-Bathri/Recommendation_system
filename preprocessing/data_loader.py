import torch
from torch_geometric.datasets import Reddit
import os
from tqdm import tqdm

def load_dataset(root, name):
    """
    Load the dataset from PyTorch Geometric
    
    Args:
        root (str): Root directory where the dataseth should be saved
        name (str): Name of the dataset
        
    Returns:
        data: PyTorch Geometric Data objecto
    """
    print(f"Loading {name} dataset...")
    if name == 'Reddit':
        dataset = Reddit(root=root)
        data = dataset[0]  # Reddit dataset typically has one graph
        
        # Debugging: Check available attributes
        print("Dataset attributes:", dir(data))
        
        #n Handle missing num_classes
        if hasattr(data, 'num_classes'):
            print(f"Number of classes: {data.num_classes}")
        else:
            print("Number of classes: Not available (num_classes attribute missing)")
            # Fallback: Infer num_classes if possible
            if hasattr(data, 'y'):
                data.num_classes = len(torch.unique(data.y))
                print(f"Inferred number of classes: {data.num_classes}")
            else:
                raise AttributeError("Cannot determine the number of classes. 'y' attribute is missing.")
        
        print(f"Dataset loaded successfully!")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges // 2}")  # Divide by 2 for undirected graphs
        print(f"Number  of node features: {data.num_node_features}")
        
        return data
    else:
        raise ValueError(f"Dataset {name} is not supported")

def load_dataset_in_chunks(root, name, save_dir='processed'):
    """
    Load the dataset and save it in chunks for processing on limited memory machines
    
    Args:
        root (str): Root directory where the dataset should be saved
        name (str): Name of the dataset
        save_dir (str): Directory to save the chunks
        
    Returns:
        data: PyTorch Geometric Data object
        chunk_info (dict): Information about the chunks
    """
    data = load_dataset(root, name)
    
    # Create directory to save chunks
    os.makedirs(save_dir, exist_ok=True)
    
    # Save node features in chunks
    chunk_size = 10000  # Adjust based on your VM memory
    num_chunks = (data.num_nodes + chunk_size - 1) // chunk_size
    
    print(f"Saving dataset in {num_chunks} chunks...")
    for i in tqdm(range(num_chunks)):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, data.num_nodes)
        
        # Save node features chunk
        if data.x is not None:
            torch.save(data.x[start_idx:end_idx], f"{save_dir}/x_chunk_{i}.pt")
        
        # Save node labels chunk if available
        if hasattr(data, 'y') and data.y is not None:
            torch.save(data.y[start_idx:end_idx], f"{save_dir}/y_chunk_{i}.pt")
    
    # Save edge information
    torch.save(data.edge_index, f"{save_dir}/edge_index.pt")
    
    # Save dataset information
    chunk_info = {
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'num_features': data.num_node_features,
        'num_chunks': num_chunks,
        'chunk_size': chunk_size
    }
    
    torch.save(chunk_info, f"{save_dir}/chunk_info.pt")
    print(f"Dataset saved in chunks successfully!")
    
    return data, chunk_info

def load_processed_chunks(processed_dir='processed'):
    """
    Load the processed dataset chunks
    
    Args:
        processed_dir (str): Directory where chunks are saved
        
    Returns:
        chunk_info (dict): Information about the chunks
    """
    chunk_info = torch.load(f"{processed_dir}/chunk_info.pt")
    print(f"Loaded chunk information:")
    print(f"Number of nodes: {chunk_info['num_nodes']}")
    print(f"Number of edges: {chunk_info['num_edges']}")
    print(f"Number of features: {chunk_info['num_features']}")
    print(f"Number of chunks: {chunk_info['num_chunks']}")
    
    return chunk_info