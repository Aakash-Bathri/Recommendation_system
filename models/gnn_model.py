# gnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphNorm

class ImprovedGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.norm1 = GraphNorm(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        self.norm2 = GraphNorm(hidden_channels)
        self.conv3 = GATConv(hidden_channels, out_channels, heads=1)
        
        self.res_fc = nn.Linear(in_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        res = self.res_fc(x)
        x = F.elu(self.norm1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.norm2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv3(x, edge_index) + res

class BatchGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 heads=4, dropout=0.3):  # Added heads parameter
        super().__init__()
        self.standard_model = ImprovedGNN(
            in_channels, 
            hidden_channels, 
            out_channels,
            heads=heads,  # Pass heads to ImprovedGNN
            dropout=dropout
        )
        self.heads = heads  # Store heads parameter
    
    def forward(self, x, edge_index, batch_size=None):
        if batch_size is None or x.size(0) <= batch_size:
            return self.standard_model(x, edge_index)
        
        embeddings = []
        num_nodes = x.size(0)
        num_batches = (num_nodes + batch_size - 1) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = min((i+1)*batch_size, num_nodes)
            
            # Extract subgraph edges with multi-head compatibility
            node_mask = (edge_index[0] >= start) & (edge_index[0] < end) & \
                        (edge_index[1] >= start) & (edge_index[1] < end)
            batch_edges = edge_index[:, node_mask] - start
            
            if batch_edges.size(1) == 0:
                # Handle empty edge case with residual connection
                embeddings.append(self.standard_model.res_fc(x[start:end]))
                continue
                
            batch_x = x[start:end]
            batch_out = self.standard_model(batch_x, batch_edges)
            embeddings.append(batch_out)

        return torch.cat(embeddings, dim=0)