'''
Here some nice plot function can be put in.
Chat very good 
'''

import matplotlib.pyplot as plt
import os

def calculate_r2(true_labels, predictions):
    # Calculate the mean of true labels
    mean_true_labels = sum(true_labels) / len(true_labels)
    
    # Calculate the total sum of squares (TSS) and residual sum of squares (RSS)
    tss = sum((y - mean_true_labels) ** 2 for y in true_labels)
    rss = sum((y - y_pred) ** 2 for y, y_pred in zip(true_labels, predictions))
    
    # Calculate R²
    r2 = 1 - (rss / tss)
    return r2


def simple_loss_plot(losses, val_loss):
    """
    Plots the loss values over epochs.

    Parameters:
    losses (list or array-like): A list or array of loss values.

    Returns:
    None
    """

    os.makedirs('output', exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.plot(val_loss, marker='o', linestyle='-', color='r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/loss_epoch_graph.png')
    plt.show()

def true_pred_plot(true_labels, predictions):
    # Calculate R² value
    r2 = calculate_r2(true_labels, predictions)
    
    # Create a scatter plot
    plt.scatter(true_labels, predictions, label='Predictions', alpha=0.5)
    
    # Add a line y=x for reference
    min_val = min(min(true_labels), min(predictions))
    max_val = max(max(true_labels), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    
    # Label the axes
    plt.xlabel('True Labels')
    plt.ylabel('Predictions')
    
    # Add a title with R² value
    plt.title(f'True Labels vs Predictions (R² = {r2:.2f})')
    
    # Add a legend
    plt.legend()
    plt.savefig('output/true_pred_graph.png')
    # Display the plot
    plt.show()