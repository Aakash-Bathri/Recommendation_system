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
from models.gnn_model import BatchGNN
from models.link_decoder import EnhancedDecoder
from training.trainer import AdvancedTrainer
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
    parser.add_argument('--pos_weight', type=float, default=5.0,
                       help='Class weighting for positive samples')
    parser.add_argument('--max_lr', type=float, default=0.005,
                       help='Maximum learning rate for OneCycle policy')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping threshold')
    return parser.parse_args()

def initialize_model(args, chunk_info):
    """Initialize model and decoder with improved architectures"""
    # Use improved components
    model = BatchGNN(
        in_channels=603,  # Verify this matches your actual feature dimensions
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        heads=4,  # Add multi-head attention
        dropout=0.3
    )
    
    decoder = EnhancedDecoder(  # Use enhanced decoder
        in_channels=args.out_channels,
        hidden_dim=512,  # Match paper's hidden dimensions
        dropout=0.4
    )
    
    return model, decoder

def train_model(args, train_data, val_data):
    import os
    # Load chunk info to get feature dimensions
    chunk_info = load_processed_chunks(args.data_dir)
    
    # Initialize model and optimizer
    model, decoder = initialize_model(args, chunk_info)
    # Initialize trainer with advanced features
    trainer = AdvancedTrainer(  # Use AdvancedTrainer instead of ModelTrainer
        model=model,
        decoder=decoder,
        device=config.DEVICE,
        checkpoint_dir=args.checkpoint_dir,
        pos_weight=5.0  # Adjust based on your class imbalance
    )
    
    # Remove manual epoch loop - use built-in train() method
    trainer.train(
        train_data,
        val_data,
        epochs=args.epochs,
        batch_mode=True,
        eval_batch_size=args.batch_size,
        early_stop=args.patience
    )
    
    trainer.plot_training_history(save_dir=args.plot_dir)
    
    return model, decoder, trainer.best_auc

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
        trainer = AdvancedTrainer(model, decoder, device=config.DEVICE)
        trainer.load_checkpoint(f"{args.checkpoint_dir}/best_model.pth")
        
        model = trainer.model
        decoder = trainer.decoder
    
    if args.phase in ['evaluate', 'all']:
        test_metrics = evaluate_model(args, model, decoder, test_data)
    
    print("\n=== Complete ===")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    main()