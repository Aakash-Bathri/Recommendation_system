import torch
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from tqdm import tqdm
import os
import time
from torch_geometric.utils import to_networkx

def generate_deepwalk_random_walks(G, num_walks, walk_length):
    """
    Generate random walks for DeepWalk
    
    Args:
        G (networkx.Graph): NetworkX graph
        num_walks (int): Number of walks per node
        walk_length (int): Length of each walk
        
    Returns:
        walks (list): List of random walks
    """
    walks = []
    nodes = list(G.nodes())
    
    for _ in tqdm(range(num_walks), desc="Generating walks"):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                neighbors = list(G.neighbors(walk[-1]))
                if not neighbors:
                    break
                walk.append(np.random.choice(neighbors))
            walks.append(list(map(str, walk)))
    
    return walks

def process_chunk_for_deepwalk(edge_index, node_indices, num_walks, walk_length):
    """
    Process a chunk of nodes for DeepWalk
    
    Args:
        edge_index (torch.Tensor): Edge index tensor
        node_indices (list): List of node indices to process
        num_walks (int): Number of walks per node
        walk_length (int): Length of each walk
        
    Returns:
        walks (list): List of random walks for the chunk
    """
    # Create a subgraph with only the edges connecting nodes in this chunk
    # This is a simplification - ideally we'd maintain the global graph structure
    node_set = set(node_indices)
    
    # Convert to set for faster lookup
    edge_list = []
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if src in node_set or dst in node_set:
            edge_list.append((src, dst))
    
    G = nx.Graph()
    G.add_nodes_from(node_indices)
    G.add_edges_from(edge_list)
    
    # Generate walks starting only from nodes in this chunk
    walks = []
    for _ in range(num_walks):
        for node in node_indices:
            if node not in G:  # Skip isolated nodes
                continue
            walk = [node]
            while len(walk) < walk_length:
                neighbors = list(G.neighbors(walk[-1]))
                if not neighbors:
                    break
                walk.append(np.random.choice(neighbors))
            walks.append(list(map(str, walk)))
    
    return walks

def generate_deepwalk_features_in_chunks(processed_dir, window=5, walk_length=20, num_walks=10, vector_size=128, batch_size=5000):
    """
    Generate DeepWalk features in chunks to save memory
    
    Args:
        processed_dir (str): Directory with processed chunks
        window (int): Window size for Word2Vec
        walk_length (int): Length of each walk
        num_walks (int): Number of walks per node
        vector_size (int): Size of the embedding vector
        batch_size (int): Number of nodes to process at once
        
    Returns:
        None (saves embeddings to disk)
    """
    print("Loading graph data...")
    edge_index = torch.load(f"{processed_dir}/edge_index.pt")
    chunk_info = torch.load(f"{processed_dir}/chunk_info.pt")
    
    num_nodes = chunk_info['num_nodes']
    num_chunks = chunk_info['num_chunks']
    chunk_size = chunk_info['chunk_size']
    
    print("Converting to NetworkX graph...")
    # Create a placeholder graph with all nodes
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    # Add edges from edge_index
    for i in tqdm(range(edge_index.size(1)), desc="Adding edges"):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        G.add_edge(src, dst)
    
    print(f"Generated NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Process in batches to avoid memory issues
    all_walks = []
    num_batches = (num_nodes + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_nodes)
        node_indices = list(range(start_idx, end_idx))
        
        print(f"Processing batch {batch_idx+1}/{num_batches} (nodes {start_idx}-{end_idx})")
        batch_walks = process_chunk_for_deepwalk(edge_index, node_indices, num_walks, walk_length)
        all_walks.extend(batch_walks)
        
        # Save memory by clearing unnecessary data
        if batch_idx % 5 == 0:
            print("Clearing memory...")
            import gc
            gc.collect()
    
    print(f"Total walks generated: {len(all_walks)}")
    print("Training Word2Vec model...")
    model = Word2Vec(
        all_walks,
        vector_size=vector_size,
        window=window,
        min_count=1,
        sg=1,
        workers=os.cpu_count(),
        epochs=10
    )
    
    # Save the model to disk
    model_path = f"{processed_dir}/deepwalk_model.model"
    model.save(model_path)
    print(f"Saved Word2Vec model to {model_path}")
    
    # Generate embeddings for each node in chunks to save memory
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_nodes)
        
        print(f"Generating embeddings for chunk {i+1}/{num_chunks}")
        topo_features = []
        
        for node_idx in tqdm(range(start_idx, end_idx)):
            try:
                topo_features.append(model.wv[str(node_idx)])
            except KeyError:
                # If node not in vocabulary, use a zero vector
                topo_features.append(np.zeros(vector_size))
        
        # Convert to tensor and save
        topo_features = torch.tensor(np.array(topo_features), dtype=torch.float)
        torch.save(topo_features, f"{processed_dir}/topo_features_chunk_{i}.pt")
    
    print("DeepWalk feature generation complete!")

def combine_features(processed_dir):
    """
    Combine original features with topological features
    
    Args:
        processed_dir (str): Directory with processed chunks
        
    Returns:
        None (saves combined features to disk)
    """
    chunk_info = torch.load(f"{processed_dir}/chunk_info.pt")
    num_chunks = chunk_info['num_chunks']
    
    for i in range(num_chunks):
        print(f"Processing chunk {i+1}/{num_chunks}")
        
        # Load original features
        try:
            x_chunk = torch.load(f"{processed_dir}/x_chunk_{i}.pt")
            has_original_features = True
        except FileNotFoundError:
            has_original_features = False
        
        # Load topological features
        topo_features = torch.load(f"{processed_dir}/topo_features_chunk_{i}.pt")
        
        # Combine features
        if has_original_features:
            combined_features = torch.cat([x_chunk, topo_features], dim=1)
        else:
            combined_features = topo_features
        
        # Save combined features
        torch.save(combined_features, f"{processed_dir}/combined_features_chunk_{i}.pt")
    
    print("Feature combination complete!")