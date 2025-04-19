import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data

# Import project modules
import config
from preprocessing.data_loader import load_dataset_in_chunks, load_processed_chunks
from preprocessing.feature_engineering import generate_deepwalk_features_in_chunks, combine_features
from preprocessing.data_splitter import split_and_save_data, load_split_data
from models.gnn_model import HetInfLinkPred, BatchHetInfLinkPred
from models.link_decoder import LinkDecoder
from training.trainer import ModelTrainer
from training.evaluator import evaluate_and_visualize
from utils.visualization import plot_training_history, plot_all_evaluation_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Link Prediction on Reddit Dataset')
    parser.add_argument('--phase', type=str, default='all', 
                        choices=['data', 'features', 'combine', 'train', 'evaluate', 'all'],
                        help='Which phase to run')
    parser.add_argument('--data_dir', type=str, default='processed',
                        help='Directory for processed data')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--hidden_channels', type=int, default=512,
                        help='Hidden channels for GNN')
    parser.add_argument('--out_channels', type=int, default=256,
                        help='Output channels for GNN')
    parser.add_argument