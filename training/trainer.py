import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy
from tqdm import tqdm
import numpy as np

from utils.metrics import evaluate_model_detailed

class ModelTrainer:
    """
    Trainer class for the link prediction model
    """
    def __init__(
        self, 
        model, 
        decoder, 
        optimizer=None, 
        scheduler=None,
        criterion=None,
        device='cuda',
        checkpoint_dir='checkpoints'
    ):
        self.model = model.to(device)
        self.decoder = decoder.to(device)
        self.device = device
        
        # Default optimizer if none provided
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                list(model.parameters()) + list(decoder.parameters()),
                lr=0.001,
                weight_decay=5e-4
            )
        else:
            self.optimizer = optimizer
        
        # Default loss function if none provided
        if criterion is None:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = criterion
        
        # Learning rate scheduler
        self.scheduler = scheduler
        
        # Create checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training metrics
        self.train_losses = []
        self.val_metrics = []
        self.best_val_metric = 0
        self.epochs_no_improve = 0
    
    def train_epoch(self, data):
        """
        Train for a single epoch
        
        Args:
            data: Training data
            
        Returns:
            float: Training loss
        """
        self.model.train()
        self.decoder.train()
        
        # Move data to device
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_label_index = data.edge_label_index.to(self.device)
        edge_label = data.edge_label.float().to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        z = self.model(x, edge_index)
        pred = self.decoder(z, edge_label_index)
        loss = self.criterion(pred, edge_label)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch_batched(self, data, batch_size=128):
        """
        Train for a single epoch with both node and edge batching for memory efficiency.

        Args:
            data: Training data
            batch_size: Batch size for edge processing

        Returns:
            float: Average training loss
        """
        self.model.train()
        self.decoder.train()

        # Move node features and edges to device
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_label_index = data.edge_label_index
        edge_label = data.edge_label.float()

        # Generate node embeddings (supporting node-level batching, if applicable)
        z = self.model(x, edge_index, batch_size=batch_size).detach()

        total_loss = 0.0
        num_batches = 0
        num_edges = edge_label_index.size(1)

        for i in range(0, num_edges, batch_size):
            edge_batch = edge_label_index[:, i:i + batch_size].to(self.device)
            label_batch = edge_label[i:i + batch_size].to(self.device)

            self.optimizer.zero_grad()
            pred = self.decoder(z, edge_batch)
            loss = self.criterion(pred, label_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            print(f"Processing edge batch {i//batch_size + 1}")


        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def load_checkpoint(self, path):
        """
        Load model and optimizer from checkpoint.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        print(f"Loaded checkpoint from {path}")
        
    def save_checkpoint(self, filename="model_checkpoint.pt"):
        """
        Save model, decoder, and optimizer state to a checkpoint file.
        
        Args:
            filename (str): Path to the checkpoint file
        """
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")