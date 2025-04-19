import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class HetInfLinkPred(nn.Module):
    """
    Graph Neural Network model for heterogeneous information network link prediction
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        # Semantic aggregation (MLP)
        self.semantic_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Topological aggregation (GAT)
        self.gat1 = GATConv(hidden_channels, hidden_channels//2, heads=4)
        self.gat2 = GATConv(2 * hidden_channels, out_channels, heads=2, concat=False)
        self.gat3 = GATConv(out_channels, out_channels, concat=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Edge index
            
        Returns:
            torch.Tensor: Node embeddings
        """
        # Semantic aggregation
        x_semantic = self.semantic_mlp(x)
        # Topological aggregation with GAT layers
        x = F.elu(self.gat1(x_semantic, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat3(x, edge_index)
        return x

# Model with batch processing capability for larger graphs
class BatchHetInfLinkPred(nn.Module):
    """
    GNN model with batch processing capabilities for larger graphs
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.standard_model = HetInfLinkPred(
            in_channels, hidden_channels, out_channels, dropout
        )
    
    def forward(self, x, edge_index, batch_size=None):
        """
        Forward pass with optional batching for large graphs
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Edge index
            batch_size (int): Batch size for processing (None for full graph)
            
        Returns:
            torch.Tensor: Node embeddings
        """
        if batch_size is None or x.size(0) <= batch_size:
            # Process entire graph at once
            return self.standard_model(x, edge_index)
        else:
            # Process in batches (simple approach - not ideal for GNNs but can help with memory)
            embeddings = []
            num_batches = (x.size(0) + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, x.size(0))
                
                # Filter edges relevant to this batch (simplified approach)
                mask = (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)
                batch_edges = edge_index[:, mask]
                
                # Adjust edge indices to be relative to the batch
                batch_edges[0] = batch_edges[0] - start_idx
                batch_edges[1] = torch.clamp(batch_edges[1], min=0, max=end_idx-1)
                
                batch_x = x[start_idx:end_idx]
                batch_embeddings = self.standard_model(batch_x, batch_edges)
                embeddings.append(batch_embeddings)
            
            return torch.cat(embeddings, dim=0)