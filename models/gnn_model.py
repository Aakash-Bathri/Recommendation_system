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
            num_nodes = x.size(0)
            num_batches = (num_nodes + batch_size - 1) // batch_size

            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, num_nodes)

                # Select edges where both source and target are within this batch
                node_mask = (edge_index[0] >= start) & (edge_index[0] < end) & \
                            (edge_index[1] >= start) & (edge_index[1] < end)
                batch_edges = edge_index[:, node_mask]

                if batch_edges.size(1) == 0:
                    # No valid edges in this batch, skip
                    continue

                # Re-index edge_index to batch-relative indices
                batch_edges = batch_edges - start  # Shift node indices to start from 0

                batch_x = x[start:end]
                batch_out = self.standard_model(batch_x, batch_edges)
                embeddings.append(batch_out)

            return torch.cat(embeddings, dim=0)