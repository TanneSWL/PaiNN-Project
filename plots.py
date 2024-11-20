'''
Here some nice plot function can be put in.
Chat very good 
'''

import matplotlib.pyplot as plt

def simple_loss_plot(losses):
    """
    Plots the loss values over epochs.

    Parameters:
    losses (list or array-like): A list or array of loss values.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, marker='o', linestyle='-', color='b', label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()