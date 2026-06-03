import torch 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from statsmodels.stats.contingency_tables import mcnemar
import torch


EMOTION_CLASSES = [
    'amusement', 'anger', 'awe', 'contentment',
    'disgust', 'excitement', 'fear', 'sadness'
]

def evaluate(model, loader, criterion, device):
    """Basic evaluation loop, returns loss and accuracy."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

def get_predictions(model, loader, device):
    """Get predictions and probabilities from a PyTorch model."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    return (
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_labels)
    )
    
def compute_metrics(labels, preds, probs):
    """Compute F1-macro and AUC-ROC."""
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    return {'f1_macro': f1, 'auc_roc': auc}


def print_classification_report(labels, preds):
    """Print per-class classification report."""
    print(classification_report(labels, preds,
                                target_names=EMOTION_CLASSES,
                                zero_division=0))


def plot_training_curves(histories, save_path=None):
    """Plot loss and accuracy curves for multiple models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'ViT': ('#1E2761', '#028090'),
              'ResNet-50': ('#B85042', '#E7A97A')}

    for name, history in histories.items():
        c_train, c_val = colors.get(name, ('#333333', '#999999'))
        style = '--' if name == 'ResNet-50' else '-'
        axes[0].plot(history['train_loss'], label=f'{name} Train',
                     color=c_train, linestyle=style)
        axes[0].plot(history['val_loss'], label=f'{name} Val',
                     color=c_val, linestyle=style)
        axes[1].plot(history['train_acc'], label=f'{name} Train',
                     color=c_train, linestyle=style)
        axes[1].plot(history['val_acc'], label=f'{name} Val',
                     color=c_val, linestyle=style)

    axes[0].set_title('Loss Curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].set_title('Accuracy Curves')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_benchmark_comparison(results, save_path=None):
    """Bar chart comparing all models on val acc, F1, AUC-ROC."""
    model_names = list(results.keys())
    colors = ['#1E2761', '#028090', '#02C39A', '#B85042']
    metrics = ['val_acc', 'f1_macro', 'auc_roc']
    titles = ['Validation Accuracy', 'F1-Macro', 'AUC-ROC (macro)']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric, title in zip(axes, metrics, titles):
        values = [results[m].get(metric, 0) for m in model_names]
        short_names = [n.replace(' ', '\n') for n in model_names]
        bars = ax.bar(short_names, values, color=colors, width=0.5)
        ax.set_title(title)
        ax.set_ylim(0, max(values) * 1.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f'{val:.3f}', ha='center', fontweight='bold')

    plt.suptitle('Benchmark Comparison: All Models',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrices(labels, predictions_dict, save_path=None):
    """Plot normalised confusion matrices for given models."""
    n = len(predictions_dict)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, preds) in zip(axes, predictions_dict.items()):
        cm = confusion_matrix(labels, preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', ax=ax,
                    xticklabels=EMOTION_CLASSES,
                    yticklabels=EMOTION_CLASSES,
                    cmap='Blues')
        ax.set_title(f'{name}:  Normalised Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def run_mcnemar_test(labels, preds_a, preds_b, name_a='ViT', name_b='ResNet-50'):
    """Run McNemar's test between two models."""
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)

    contingency = np.array([
        [np.sum(correct_a & correct_b), c],
        [b, np.sum(~correct_a & ~correct_b)]
    ])

    result = mcnemar(contingency, exact=False, correction=True)

    print(f"McNemar's Test: {name_a} vs {name_b}")
    print(f"Contingency table:\n{contingency}")
    print(f"Statistic: {result.statistic:.4f}")
    print(f"p-value:   {result.pvalue:.4f}")
    if result.pvalue < 0.05:
        print("Statistically significant difference (p < 0.05)")
    else:
        print("No statistically significant difference (p >= 0.05)")

    return result