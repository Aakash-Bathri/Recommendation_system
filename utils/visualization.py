import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from matplotlib.ticker import MaxNLocator
import os

def set_plotting_style():
    """
    Set consistent plotting style
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set(font_scale=1.2)
    colors = sns.color_palette("Set2")
    return colors

def plot_training_history(train_losses, val_metrics, save_dir='plots'):
    """
    Plot training loss and validation metrics
    
    Args:
        train_losses (list): List of training losses
        val_metrics (list): List of validation metric dictionaries
        save_dir (str): Directory to save plots
    """
    colors = set_plotting_style()
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract validation AUC and AP
    val_aucs = [m['auc'] for m in val_metrics]
    val_aps = [m['ap'] for m in val_metrics] if 'ap' in val_metrics[0] else None
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color=colors[0])
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_loss.png', dpi=300)
    plt.close()

    # Plot validation AUC
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(val_aucs) + 1), val_aucs, marker='s', linestyle='-', color=colors[1])
    plt.title('Validation AUC Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('AUC Score')
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(f'{save_dir}/validation_auc.png', dpi=300)
    plt.close()
    
    # Plot validation AP if available
    if val_aps:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(val_aps) + 1), val_aps, marker='d', linestyle='-', color=colors[2])
        plt.title('Validation Average Precision Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('AP Score')
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(f'{save_dir}/validation_ap.png', dpi=300)
        plt.close()
    
    # Combined plot of metrics
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', label='Training Loss', color=colors[0])
    
    # Normalize metrics for better visualization
    max_loss = max(train_losses)
    normalized_aucs = [auc * max_loss for auc in val_aucs]
    
    plt.plot(range(1, len(val_aucs) + 1), normalized_aucs, marker='s', linestyle='-', label='Validation AUC (scaled)', color=colors[1])
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_progress.png', dpi=300)
    plt.close()

def plot_roc_curve(true_labels, pred_probs, auc, save_dir='plots'):
    """
    Plot ROC curve
    
    Args:
        true_labels (np.ndarray): True labels
        pred_probs (np.ndarray): Predicted probabilities
        auc (float): AUC score
        save_dir (str): Directory to save plots
    """
    colors = set_plotting_style()
    os.makedirs(save_dir, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color=colors[2], lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/roc_curve.png', dpi=300)
    plt.close()

def plot_pr_curve(true_labels, pred_probs, ap, save_dir='plots'):
    """
    Plot Precision-Recall curve
    
    Args:
        true_labels (np.ndarray): True labels
        pred_probs (np.ndarray): Predicted probabilities
        ap (float): Average Precision score
        save_dir (str): Directory to save plots
    """
    colors = set_plotting_style()
    os.makedirs(save_dir, exist_ok=True)
    
    precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color=colors[3], lw=2, label=f'PR curve (AP = {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pr_curve.png', dpi=300)
    plt.close()

def plot_prediction_distribution(pred_probs, true_labels, save_dir='plots'):
    """
    Plot distribution of predicted probabilities
    
    Args:
        pred_probs (np.ndarray): Predicted probabilities
        true_labels (np.ndarray): True labels
        save_dir (str): Directory to save plots
    """
    colors = set_plotting_style()
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    df = pd.DataFrame({
        'Probability': pred_probs,
        'True Label': true_labels
    })
    sns.histplot(data=df, x='Probability', hue='True Label', bins=30, alpha=0.7)
    plt.title('Distribution of Prediction Probabilities')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/prediction_distribution.png', dpi=300)
    plt.close()

def plot_confusion_matrix(cm, threshold, save_dir='plots'):
    """
    Plot confusion matrix
    
    Args:
        cm (np.ndarray): Confusion matrix
        threshold (float): Decision threshold
        save_dir (str): Directory to save plots
    """
    colors = set_plotting_style()
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix (Threshold = {threshold:.2f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300)
    plt.close()

def plot_all_evaluation_metrics(metrics, save_dir='plots'):
    """
    Plot all evaluation metrics
    
    Args:
        metrics (dict): Dictionary of evaluation metrics
        save_dir (str): Directory to save plots
    """
    plot_roc_curve(metrics['true_labels'], metrics['pred_probs'], metrics['auc'], save_dir)
    plot_pr_curve(metrics['true_labels'], metrics['pred_probs'], metrics['ap'], save_dir)
    plot_prediction_distribution(metrics['pred_probs'], metrics['true_labels'], save_dir)
    plot_confusion_matrix(metrics['confusion_matrix'], metrics['optimal_threshold'], save_dir)