import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import confusion_matrix

def calculate_metrics(pred_probs, true_labels):
    """
    Calculate evaluation metrics for link prediction
    
    Args:
        pred_probs (np.ndarray): Predicted probabilities
        true_labels (np.ndarray): True labels
        
    Returns:
        dict: Dictionary of metrics
    """
    # AUC and AP
    auc = roc_auc_score(true_labels, pred_probs)
    ap = average_precision_score(true_labels, pred_probs)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
    
    # Find optimal threshold using Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Compute predictions using optimal threshold
    y_pred = (pred_probs >= optimal_threshold).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'auc': auc,
        'ap': ap,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'optimal_threshold': optimal_threshold,
        'confusion_matrix': cm,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }

def evaluate_model_detailed(model, decoder, data, device, batch_size=128):
    """
    Evaluate the model with detailed metrics
    
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

        # Generate node embeddings using batch-capable model
        z = model(x, edge_index, batch_size=batch_size)
        
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
        metrics = calculate_metrics(pred_probs, true_labels)
        metrics['pred_probs'] = pred_probs
        metrics['true_labels'] = true_labels
        
        return metrics