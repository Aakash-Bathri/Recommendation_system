import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

class AdvancedTrainer:
    def __init__(self, model, decoder, optimizer=None, device='cuda',
                 checkpoint_dir='checkpoints', pos_weight=5.0):
        self.model = model.to(device)
        self.decoder = decoder.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize optimizer with default parameters
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                list(model.parameters()) + list(decoder.parameters()),
                lr=0.001,
                weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer

        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=3, factor=0.5
        )

        # Loss function with class balancing
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )

        # Training metrics storage
        self.train_losses = []
        self.val_metrics = {'loss': [], 'auc': [], 'ap': []}
        self.best_epoch = 1
        self.early_stop_counter = 0
        self.best_auc = 0
        self.epoch = 1

    def train_epoch(self, data):
        """Train for one epoch (full graph)"""
        self.model.train()
        self.decoder.train()
        
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_label_index = data.edge_label_index.to(self.device)
        edge_label = data.edge_label.float().to(self.device)

        self.optimizer.zero_grad()
        
        # Forward pass
        z = self.model(x, edge_index)
        pred = self.decoder(z, edge_label_index)
        
        # Calculate loss
        loss = self.criterion(pred, edge_label)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train_epoch_batched(self, data, node_batch_size=2048, edge_batch_size=51200):
        """Train with batched node processing and edge sampling"""
        self.model.train()
        self.decoder.train()
        
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_label_index = data.edge_label_index
        edge_label = data.edge_label.float()

        # Generate node embeddings in batches
        with torch.no_grad():  # Freeze embeddings during edge training
            z = self.model(x, edge_index, batch_size=node_batch_size)

        # Shuffle edges for batched training
        num_edges = edge_label_index.size(1)
        perm = torch.randperm(num_edges)
        edge_label_index = edge_label_index[:, perm]
        edge_label = edge_label[perm]

        total_loss = 0
        num_batches = (num_edges + edge_batch_size - 1) // edge_batch_size
        pbar = tqdm(total=num_batches, desc="Processing edge batches")
        for i in range(num_batches):
            start_idx = i * edge_batch_size
            end_idx = min((i+1)*edge_batch_size, num_edges)
            
            batch_edges = edge_label_index[:, start_idx:end_idx].to(self.device)
            batch_labels = edge_label[start_idx:end_idx].to(self.device)

            self.optimizer.zero_grad()
            
            # Edge prediction
            pred = self.decoder(z, batch_edges)
            loss = self.criterion(pred, batch_labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.update(1)
        pbar.close()
            

        return total_loss / num_batches

    def evaluate(self, data, batch_size=None):
        """Evaluate model on validation/test data"""
        self.model.eval()
        self.decoder.eval()
        
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_label_index = data.edge_label_index
        edge_label = data.edge_label.float()

        with torch.no_grad():
            # Generate embeddings
            z = self.model(x, edge_index, batch_size=batch_size)
            
            if batch_size is None:
                # Full graph evaluation
                pred = self.decoder(z, edge_label_index.to(self.device))
                probs = torch.sigmoid(pred).cpu()
            else:
                # Batched evaluation
                probs = []
                num_edges = edge_label_index.size(1)
                for i in range(0, num_edges, batch_size):
                    batch_edges = edge_label_index[:, i:i+batch_size].to(self.device)
                    batch_pred = self.decoder(z, batch_edges)
                    probs.append(torch.sigmoid(batch_pred).cpu())
                probs = torch.cat(probs)

        labels = edge_label.cpu().numpy()
        probs = probs.numpy()

        auc = roc_auc_score(labels, probs)
        ap = average_precision_score(labels, probs)
        loss = self.criterion(torch.tensor(probs), torch.tensor(labels)).item()

        return {'loss': loss, 'auc': auc, 'ap': ap}

    def save_checkpoint(self, filename='checkpoint.pth', best=False):
        """Save training state"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state': self.model.state_dict(),
            'decoder_state': self.decoder.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'best_auc': self.best_auc,
            'best_epoch': self.best_epoch,
            'early_stop_counter': self.early_stop_counter
        }
        
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        
        if best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, filename='checkpoint.pth'):
        """Load training state"""
        path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
            
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.decoder.load_state_dict(checkpoint['decoder_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.train_losses = checkpoint['train_losses']
        self.val_metrics = checkpoint['val_metrics']
        self.best_auc = checkpoint['best_auc']
        self.epoch = checkpoint['epoch']
        self.best_epoch = checkpoint['best_epoch']
        self.early_stop_counter = checkpoint['early_stop_counter']
        
        print(f"Loaded checkpoint from epoch {self.epoch + 1}")

    def plot_training_history(self, save_dir=None):
        """
        Plot training loss and validation metrics using class's stored history
        """
        # Set default save directory
        if save_dir is None:
            save_dir = "plots"
        os.makedirs(save_dir, exist_ok=True)

        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set(font_scale=1.2)
        colors = sns.color_palette("Set2")

        # Extract metrics from instance variables
        train_losses = self.train_losses
        val_aucs = self.val_metrics['auc']
        val_aps = self.val_metrics['ap'] if 'ap' in self.val_metrics else None

        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses)+1), train_losses, 
                marker='o', linestyle='-', color=colors[0])
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_loss.png'), dpi=300)
        plt.close()

        # Plot validation AUC
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(val_aucs)+1), val_aucs,
                marker='s', linestyle='-', color=colors[1])
        plt.title('Validation AUC Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('AUC Score')
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'validation_auc.png'), dpi=300)
        plt.close()

        # Plot validation AP if available
        if val_aps:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(val_aps)+1), val_aps,
                    marker='d', linestyle='-', color=colors[2])
            plt.title('Validation Average Precision Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('AP Score')
            plt.grid(True)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'validation_ap.png'), dpi=300)
            plt.close()

        # Combined progress plot
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(train_losses)+1), train_losses,
                marker='o', linestyle='-', label='Training Loss', color=colors[0])
        
        if train_losses:
            max_loss = max(train_losses)
            normalized_aucs = [auc * max_loss for auc in val_aucs]
            plt.plot(range(1, len(val_aucs)+1), normalized_aucs,
                    marker='s', linestyle='-', 
                    label='Validation AUC (scaled)', color=colors[1])
        
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_progress.png'), dpi=300)
        plt.close()
        
    def train(self, train_data, val_data, epochs=100, 
            batch_mode=True, eval_batch_size=None,
            save_interval=5, early_stop=10):
        """Complete training loop with proper checkpoint continuation"""
        # Check for existing checkpoint to resume training
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            print("Found existing checkpoint. Resuming training...")
            self.load_checkpoint()

        # Initialize from current state
        start_epoch = self.epoch
        end_epoch = start_epoch + epochs
        best_epoch = self.best_epoch
        early_stop_counter = self.early_stop_counter

        try:
            for epoch in range(start_epoch, end_epoch):
                # Train phase
                if batch_mode:
                    train_loss = self.train_epoch_batched(train_data)
                else:
                    train_loss = self.train_epoch(train_data)
                
                self.train_losses.append(train_loss)

                # Validation phase
                val_results = self.evaluate(val_data, batch_size=eval_batch_size)
                self.val_metrics['loss'].append(val_results['loss'])
                self.val_metrics['auc'].append(val_results['auc'])
                self.val_metrics['ap'].append(val_results['ap'])
                
                # Update learning rate
                self.scheduler.step(val_results['auc'])
                
                self.epoch = epoch + 1
                
                # Check for new best model
                if val_results['auc'] > self.best_auc:
                    self.best_auc = val_results['auc']
                    best_epoch = epoch + 1
                    self.save_checkpoint(best=True)
                    early_stop_counter = 0  # Reset counter on improvement
                else:
                    early_stop_counter += 1

                # Periodic checkpointing
                if ( (epoch + 1) % save_interval == 0 ) or epoch + 1 == end_epoch:
                    self.save_checkpoint()

                # Print progress
                print(f"Epoch {epoch+1}/{end_epoch}:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_results['loss']:.4f}")
                print(f"  Val AUC: {val_results['auc']:.4f}")
                print(f"  Val AP: {val_results['ap']:.4f}")
                print(f"  Best AUC: {self.best_auc:.4f} (epoch {best_epoch+1})")

                # Early stopping check
                if early_stop_counter >= early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving current state...")
            self.save_checkpoint("interrupted_checkpoint.pth")
            raise

        print(f"\nTraining completed. Best validation AUC: {self.best_auc:.4f} at epoch {best_epoch+1}")
        return self.best_auc