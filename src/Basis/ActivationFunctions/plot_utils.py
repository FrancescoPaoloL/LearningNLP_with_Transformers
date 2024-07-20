import numpy as np
import matplotlib.pyplot as plt

def plot_activation(ax, x, y, title):
    ax.clear()
    ax.plot(x, y)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title(title)
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.axvline(0, color='grey', linewidth=0.5)
    ax.grid(True)
    ax.legend([title])
    plt.draw()

def update_plot(event, ax, activations, label, x_values, x_softmax):
    plot_activation(ax, x_softmax if label == "Softmax Function" else x_values, activations[label], label)
