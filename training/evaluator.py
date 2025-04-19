import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

def evaluate_model(model, decoder, data, device, batch_size=None):
    """
    Evaluate the model on given data
    
    Args:
        model: GNN model
        decoder: Link decoder
        data: PyTorch Geometric Data object
        device: Device to run evaluation on
        batch_size: Batch size for evaluation (None for full batch)
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Move data to device
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        
        # Generate node embeddings
        z = model(x, edge_index)
        
        if batch_size is None:
            # Evaluate all edges at once
            edge_label_index = data.edge_label_index.to(device)
            pred = decoder(z, edge_label_index)
            pred_probs = pred.sigmoid().cpu().numpy()
        else:
            # Evaluate in batches
            edge_label_index = data.edge_label_index
            num_edges = edge_label_index.size(1)
            pred_probs = []
            
            for i in range(0, num_edges, batch_size):
                batch_edge_index = edge_label_index[:, i:i+batch_size].to(device)
                batch_pred = decoder(z, batch_edge_index)
                batch_probs = batch_pred.sigmoid().cpu().numpy()
                pred_probs.append(batch_probs)
            
            pred_probs = np.concatenate(pred_probs)
        
        # Get true labels
        true_labels = data.edge_label.cpu().numpy()
        
        # Calculate metrics
        auc = roc_auc_score(true_labels, pred_probs)
        ap = average_precision_score(true_labels, pred_probs)
        
        return {
            'auc': auc,
            'ap': ap,
            'pred_probs': pred_probs,
            'true_labels': true_labels
        }

def evaluate_and_visualize(model, decoder, test_data, device, save_dir='plots', batch_size=None):
    """
    Evaluate model and generate visualizations
    
    Args:
        model: GNN model
        decoder: Link decoder
        test_data: PyTorch Geometric Data object
        device: Device to run evaluation on
        save_dir: Directory to save plots
        batch_size: Batch size for evaluation
        
    Returns:
        dict: Evaluation metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("Evaluating model on test data...")
    metrics = evaluate_model(model, decoder, test_data, device, batch_size)
    
    # Extract metrics
    test_auc = metrics['auc']
    test_ap = metrics['ap']
    test_pred_probs = metrics['pred_probs']
    test_true_labels = metrics['true_labels']
    
    print(f"Test Results - AUC: {test_auc:.4f}, AP: {test_ap:.4f}")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(test_true_labels, test_pred_probs)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {test_auc:.3f})')
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
    
    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(test_true_labels, test_pred_probs)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {test_ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pr_curve.png', dpi=300)
    plt.close()
    
    # Plot prediction distribution
    plt.figure(figsize=(10, 8))
    sns.histplot(
        x=test_pred_probs, 
        hue=test_true_labels, 
        bins=30, 
        alpha=0.7,
        kde=True
    )
    plt.title('Distribution of Prediction Probabilities')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/prediction_distribution.png', dpi=300)
    plt.close()
    
    # Find optimal threshold using ROC curve
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = _[optimal_idx]
    
    # Calculate confusion matrix with optimal threshold
    pred_labels = (test_pred_probs >= optimal_threshold).astype(int)
    cm = confusion_matrix(test_true_labels, pred_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Threshold = {optimal_threshold:.2f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300)
    plt.close()
    
    return {
        'auc': test_auc,
        'ap': test_ap,
        'optimal_threshold': optimal_threshold,
        'confusion_matrix': cm
    }