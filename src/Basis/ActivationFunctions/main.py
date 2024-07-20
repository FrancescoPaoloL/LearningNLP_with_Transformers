import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from plot_utils import plot_activation, update_plot
from activation_dict import get_activation_functions

def main():
    x_values = np.linspace(-5, 5, 400)
    x_softmax = np.linspace(-5, 5, 10)

    activations = get_activation_functions(x_values, x_softmax)

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)

    buttons = []
    for i, label in enumerate(activations.keys()):
        ax_button = plt.axes([0.05, 0.8 - i * 0.07, 0.1, 0.05])  # Position of button
        button = Button(ax_button, label)
        button.on_clicked(lambda event, label=label: update_plot(event, ax, activations, label, x_values, x_softmax))
        buttons.append(button)

    plot_activation(ax, x_values, activations[list(activations.keys())[0]], list(activations.keys())[0])
    plt.show()

if __name__ == "__main__":
    main()
