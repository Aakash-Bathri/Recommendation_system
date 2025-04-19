import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from tqdm import tqdm
import numpy as np

from utils.metrics import evaluate_model

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
    
    def train_epoch_batched(self, data, batch_size=4096):
        """
        Train for a single epoch with edge batching for memory efficiency
        
        Args:
            data: Training data
            batch_size: Batch size for edge processing
            
        Returns:
            float: Training loss
        """
        self.model.train()
        self.decoder.train()
        
        # Move node features and edges to device once
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        
        # Generate node embeddings once
        z = self.model(x, edge_index)
        
        # Process edge labels in batches
        total_loss = 0
        num_batches = 0
        
        edge_label_index = data.edge_label_index
        edge_label = data.edge_label.float()
        
        num_edges = edge