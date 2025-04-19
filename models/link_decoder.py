import torch
import torch.nn as nn
import torch.nn.functional as F

class LinkDecoder(nn.Module):
    """
    Decoder for link prediction based on node embeddings
    """
    def __init__(self, in_channels, hidden_layers=None, dropout=0.5):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [512, 256]
        
        # Build MLP layers
        layers = []
        last_size = 2 * in_channels  # Concatenate source and target embeddings
        
        for size in hidden_layers:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_size = size
        
        layers.append(nn.Linear(last_size, 1))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z, edge_label_index):
        """
        Forward pass
        
        Args:
            z (torch.Tensor): Node embeddings
            edge_label_index (torch.Tensor): Edge indices for prediction
            
        Returns:
            torch.Tensor: Edge prediction scores
        """
        # Extract source and target node embeddings
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        
        # Concatenate and pass through decoder
        edge_features = torch.cat([src, dst], dim=1)
        return self.decoder(edge_features).squeeze()
    
    def predict_batch(self, z, edge_index_batch, batch_size=1024):
        """
        Predict links in batches to save memory
        
        Args:
            z (torch.Tensor): Node embeddings
            edge_index_batch (torch.Tensor): Edge indices for prediction
            batch_size (int): Batch size for processing
            
        Returns:
            torch.Tensor: Edge prediction scores
        """
        predictions = []
        num_edges = edge_index_batch.size(1)
        num_batches = (num_edges + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_edges)
                
                # Extract batch edges
                batch_edges = edge_index_batch[:, start_idx:end_idx]
                
                # Predict
                batch_preds = self.forward(z, batch_edges)
                predictions.append(batch_preds)
        
        return torch.cat(predictions)