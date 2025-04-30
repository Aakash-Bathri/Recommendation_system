import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
import os
import random
from torch_geometric.utils import to_networkx
from sklearn.decomposition import PCA

class Node2Vec:
    """
    Simple Node2Vec implementation without relying on gensim
    """
    def __init__(self, G, dimensions=128, walk_length=20, num_walks=10, p=1, q=1):
        self.G = G
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p  # Return parameter
        self.q = q  # In-out parameter
        
    def _preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding random walks
        """
        alias_nodes = {}
        for node in self.G.nodes():
            unnormalized_probs = [1.0 for _ in self.G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = self._alias_setup(normalized_probs)
            
        return alias_nodes
    
    def _alias_setup(self, probs):
        """
        Compute alias nodes for sampling
        """
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)
        
        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K*prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
                
        # Process until the stacks are empty
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            
            J[small] = large
            q[large] = q[large] - (1.0 - q[small])
            
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
                
        return J, q
    
    def _alias_draw(self, J, q):
        """
        Draw sample from a non-uniform discrete distribution using alias sampling
        """
        K = len(J)
        kk = int(np.floor(np.random.rand()*K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]
    
    def generate_walks(self):
        """
        Generate random walks starting from each node
        """
        alias_nodes = self._preprocess_transition_probs()
        walks = []
        
        nodes = list(self.G.nodes())
        for _ in tqdm(range(self.num_walks), desc="Generating walks"):
            random.shuffle(nodes)
            for node in nodes:
                walk = [node]
                neighbors = list(self.G.neighbors(node))
                if not neighbors:
                    continue
                    
                for _ in range(self.walk_length - 1):
                    curr = walk[-1]
                    neighbors = list(self.G.neighbors(curr))
                    if not neighbors:
                        break
                    walk.append(random.choice(neighbors))  # Simplified: use random choice
                
                walks.append(list(map(str, walk)))
        
        return walks

def learn_embeddings(walks, dimensions=128):
    """
    Simple SVD-based approach to learn embeddings from co-occurrence matrix
    """
    print("Creating co-occurrence matrix...")
    nodes = set()
    for walk in walks:
        for node in walk:
            nodes.add(node)
    
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    cooccurrence_matrix = np.zeros((len(nodes), len(nodes)))
    
    for walk in tqdm(walks, desc="Building co-occurrence"):
        for i, node in enumerate(walk):
            # Consider neighbors within a window
            window_size = 5
            start = max(0, i - window_size)
            end = min(len(walk), i + window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    cooccurrence_matrix[node_to_idx[walk[i]], node_to_idx[walk[j]]] += 1
    
    print("Computing SVD...")
    # Apply SVD to get node embeddings
    try:
        U, _, _ = np.linalg.svd(cooccurrence_matrix, full_matrices=False)
        embeddings = {node: U[node_to_idx[node], :dimensions] for node in nodes}
    except:
        print("SVD failed, using PCA instead")
        # Fallback to PCA if SVD fails
        pca = PCA(n_components=dimensions)
        embeddings_matrix = pca.fit_transform(cooccurrence_matrix)
        embeddings = {node: embeddings_matrix[node_to_idx[node]] for node in nodes}
    
    return embeddings

def generate_graph_features(processed_dir, vector_size=128, batch_size=5000):
    """
    Generate graph-based node features without relying on gensim
    
    Args:
        processed_dir (str): Directory with processed chunks
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
    
    # Add edges from edge_index (process in batches to save memory)
    edge_batch_size = 1000000  # Process 1 million edges at a time
    for i in range(0, edge_index.size(1), edge_batch_size):
        end_idx = min(i + edge_batch_size, edge_index.size(1))
        batch_edges = []
        for j in range(i, end_idx):
            src, dst = edge_index[0, j].item(), edge_index[1, j].item()
            batch_edges.append((src, dst))
        
        print(f"Adding edges {i} to {end_idx}...")
        G.add_edges_from(batch_edges)
        
        # Clear memory
        import gc
        gc.collect()
    
    print(f"Generated NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Use simplified node2vec to generate walks
    print("Generating random walks...")
    n2v = Node2Vec(G, dimensions=vector_size, walk_length=10, num_walks=5)
    walks = n2v.generate_walks()
    
    print(f"Generated {len(walks)} walks")
    print("Learning node embeddings...")
    embeddings = learn_embeddings(walks, dimensions=vector_size)
    
    # Generate node features in chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_nodes)
        
        print(f"Processing node embeddings for chunk {i+1}/{num_chunks}")
        features = []
        
        for node_idx in tqdm(range(start_idx, end_idx)):
            node_str = str(node_idx)
            if node_str in embeddings:
                features.append(embeddings[node_str])
            else:
                # For nodes without embeddings, use zeros
                features.append(np.zeros(vector_size))
        
        # Convert to tensor and save
        topo_features = torch.tensor(np.array(features), dtype=torch.float)
        torch.save(topo_features, f"{processed_dir}/topo_features_chunk_{i}.pt")
    
    print("Node embedding generation complete!")

def generate_simple_features(processed_dir):
    """
    Generate simple node features based on graph structure
    - Much faster and memory efficient than embeddings
    - Uses degree centrality as features
    
    Args:
        processed_dir (str): Directory with processed chunks
        
    Returns:
        None (saves embeddings to disk)
    """
    print("Loading graph data...")
    edge_index = torch.load(f"{processed_dir}/edge_index.pt")
    chunk_info = torch.load(f"{processed_dir}/chunk_info.pt")
    
    num_nodes = chunk_info['num_nodes']
    num_chunks = chunk_info['num_chunks']
    chunk_size = chunk_info['chunk_size']
    
    # Create a degree dictionary (much more memory efficient than full graph)
    print("Computing node degrees...")
    degree_dict = {}
    for i in tqdm(range(edge_index.size(1))):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        
        # Update degree counts
        if src not in degree_dict:
            degree_dict[src] = 0
        if dst not in degree_dict:
            degree_dict[dst] = 0
            
        degree_dict[src] += 1
        degree_dict[dst] += 1
    
    # Find max degree for normalization
    max_degree = max(degree_dict.values()) if degree_dict else 1
    
    # Generate normalized degree features in chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_nodes)
        
        print(f"Generating degree features for chunk {i+1}/{num_chunks}")
        features = []
        
        for node_idx in tqdm(range(start_idx, end_idx)):
            # Get node degree (normalized)
            degree = degree_dict.get(node_idx, 0) / max_degree
            
            # Create a simple 1D feature
            features.append([degree])
        
        # Convert to tensor and save
        topo_features = torch.tensor(features, dtype=torch.float)
        torch.save(topo_features, f"{processed_dir}/topo_features_chunk_{i}.pt")
    
    print("Simple feature generation complete!")

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