import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from tqdm import tqdm

# Import project modules
import config
from preprocessing.data_loader import load_dataset_in_chunks, load_processed_chunks
from preprocessing.feature_engineering import generate_graph_features, combine_features
from preprocessing.data_splitter import split_and_save_data, load_split_data
from models.gnn_model import HetInfLinkPred, BatchHetInfLinkPred
from models.link_decoder import LinkDecoder
from training.trainer import ModelTrainer
from training.evaluator import evaluate_and_visualize
from utils.metrics import evaluate_model_detailed
from utils.visualization import plot_training_history, plot_all_evaluation_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Link Prediction on Reddit Dataset')
    parser.add_argument('--phase', type=str, default='all', 
                       choices=['train', 'evaluate', 'all'],
                       help='Which phase to run')
    parser.add_argument('--data_dir', type=str, default='processed',
                       help='Directory for processed data')
    parser.add_argument('--batch_size', type=int, default=8192,
                       help='Batch size for training/evaluation')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=0.005,
                       help='Learning rate')
    parser.add_argument('--hidden_channels', type=int, default=512,
                       help='Hidden channels for GNN')
    parser.add_argument('--out_channels', type=int, default=256,
                       help='Output channels for GNN')
    parser.add_argument('--plot_dir', type=str, default='plots',
                       help='Directory to save plots')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    return parser.parse_args()

def initialize_model(args, chunk_info):
    """Initialize model and decoder"""
    model = BatchHetInfLinkPred(
        
        in_channels=603,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        dropout=0.5
    )
    
    decoder = LinkDecoder(
        in_channels=args.out_channels,
        hidden_layers=[512, 256],
        dropout=0.3
    )
    
    return model, decoder

def train_model(args, train_data, val_data):
    """Train the model with evaluation"""
    import os
    # Load chunk info to get feature dimensions
    chunk_info = load_processed_chunks(args.data_dir)
    
    # Initialize model and optimizer
    model, decoder = initialize_model(args, chunk_info)
    
    trainer = ModelTrainer(
        model=model,
        decoder=decoder,
        device=config.DEVICE,
        checkpoint_dir=args.checkpoint_dir
    )

    # Load from checkpoint if it exists
    checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    best_metrics = None
    train_losses = []
    val_metrics = []

    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
        train_losses = trainer.train_losses
        val_metrics = trainer.val_metrics
        if val_metrics:
            best_metrics = max(val_metrics, key=lambda m: m['auc'])

    print(f"\n=== Training for {args.epochs} epochs ===")
    for epoch in tqdm(range(1, args.epochs + 1)):
        # Train epoch
        loss = trainer.train_epoch_batched(train_data, batch_size=args.batch_size)
        train_losses.append(loss)
        
        # Validate every 2 epochs
        if epoch % 2 == 0 or epoch == args.epochs:
            metrics = evaluate_model_detailed(
                model, decoder, val_data, config.DEVICE, args.batch_size
            )
            val_metrics.append(metrics)
            
            print(f"\nEpoch {epoch:03d}:")
            print(f"Train Loss: {loss:.4f}")
            print(f"Val AUC: {metrics['auc']:.4f}, Val AP: {metrics['ap']:.4f}")
            
            # Save best model
            if best_metrics is None or metrics['auc'] > best_metrics['auc']:
                best_metrics = metrics
                trainer.train_losses = train_losses
                trainer.val_metrics = val_metrics
                trainer.save_checkpoint(f"best_model.pt")
                print("Saved new best model")
                
            # Early stopping check
            if epoch > args.patience:
                recent_auc = [m['auc'] for m in val_metrics[-args.patience:]]
                if max(recent_auc) < best_metrics['auc']:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    # Plot training history
    plot_training_history(train_losses, val_metrics, save_dir=args.plot_dir)
    
    return model, decoder, best_metrics


def evaluate_model(args, model, decoder, test_data):
    """Evaluate model on test set"""
    print("\n=== Evaluating on Test Set ===")
    test_metrics = evaluate_model_detailed(
        model, decoder, test_data, config.DEVICE, args.batch_size
    )
    
    print("\nTest Metrics:")
    print(f"AUC: {test_metrics['auc']:.4f}")
    print(f"AP: {test_metrics['ap']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    
    # Generate evaluation plots
    plot_all_evaluation_metrics(test_metrics, save_dir=args.plot_dir)
    
    return test_metrics

def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load split data
    train_data, val_data, test_data = load_split_data(args.data_dir)
    
    if args.phase in ['train', 'all']:
        model, decoder, val_metrics = train_model(args, train_data, val_data)
    else:
        # Load best model for evaluation
        chunk_info = load_processed_chunks(args.data_dir)
        model, decoder = initialize_model(args, chunk_info)
        trainer = ModelTrainer(model, decoder, device=config.DEVICE)
        trainer.load_checkpoint(f"{args.checkpoint_dir}/best_model.pt")
    
    if args.phase in ['evaluate', 'all']:
        test_metrics = evaluate_model(args, model, decoder, test_data)
    
    print("\n=== Complete ===")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    main()