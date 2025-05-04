# link_decoder.py
import torch
import torch.nn as nn

class EnhancedDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=512, dropout=0.4):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(4 * in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.PReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        
    def forward(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        edge_features = torch.cat([src, dst, src * dst, torch.abs(src - dst)], dim=1)
        return self.decoder(edge_features).squeeze()
    
    def predict_batch(self, z, edge_index_batch, batch_size=1024):
        predictions = []
        num_edges = edge_index_batch.size(1)
        
        with torch.no_grad():
            for i in range(0, num_edges, batch_size):
                batch_edges = edge_index_batch[:, i:i+batch_size]
                src = z[batch_edges[0]]
                dst = z[batch_edges[1]]
                features = torch.cat([src, dst, src*dst, torch.abs(src-dst)], dim=1)
                predictions.append(self.decoder(features).squeeze().cpu())
        
        return torch.cat(predictions)