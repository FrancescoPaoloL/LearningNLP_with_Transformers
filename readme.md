# Learning NLP with Python

## Basis

### The Perceptron
We start with the simplest form of a neural network, a perceptron. It consists of a single neuron that takes input, applies weights, and outputs a result based on an activation function. This basic unit learns to classify inputs into two categories by adjusting its weights during training.
In order to show how it works, I've written a simple code that uses a "perceptron" which is implemented implicitly within the calculation of the output during the forward propagation step and the subsequent adjustment of synaptic weights during the backpropagation step.

### Activation Functions
There are various activation functions used in neural networks.

#### **1. Threshold Functions**

##### Step Function

The **Step Function** outputs 1 if the input is 0 or positive, and 0 if the input is negative. It's useful for binary decisions.

$$
f(x) = \begin{cases} 
1 & \text{if } x \geq 0 \\
0 & \text{if } x < 0 
\end{cases}
$$

#### **2. Linear Functions**

##### Piecewise Linear Function

The **Piecewise Linear Function** changes behavior at different input ranges: outputs -0.5 for inputs less than -1, passes inputs through unchanged between -1 and 1, and outputs 0.5 for inputs greater than 1.

$$
f(x) = \begin{cases} 
-0.5 & \text{if } x < -1 \\
x & \text{if } -1 \leq x < 1 \\
0.5 & \text{if } x \geq 1 
\end{cases}
$$

##### Maxout ReLU

The **Maxout ReLU** takes the maximum of the input and half the input. This can help models learn more complex patterns.

$$
f(x) = \max(x, 0.5 \cdot x)
$$

#### **3. Parametric Functions**

##### Parametric ReLU (PReLU)

The **Parametric ReLU** allows the negative slope to be learned, rather than being fixed. It’s useful for allowing the network to adapt better.

$$
f(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0 
\end{cases}
$$

where $\alpha$ is a small constant (like 0.01).

##### Exponential ReLU

The **Exponential ReLU** uses an exponential function for negative inputs and is linear for positive inputs. This helps in avoiding zero gradients for negative values.

$$
f(x) = \begin{cases} 
x & \text{if } x > 0 \\
e^x - 1 & \text{if } x \leq 0 
\end{cases}
$$

#### **4. Smooth Functions**

##### Softplus

The **Softplus** function smooths the ReLU function. It’s continuous and differentiable everywhere, which helps in training neural networks.

$$
f(x) = \log(1 + e^x)
$$

##### Swish

The **Swish** function is a smooth, self-gated activation function that tends to work well in deep networks. It helps gradients flow better during training.

$$
f(x) = \frac{x}{1 + e^{-\beta x}}
$$

where $\beta$ is a parameter (often 1).

#### **5. Normalization Functions**

##### Sigmoid

The **Sigmoid** function compresses input values to a range between 0 and 1, which is useful for probability estimates in binary classification.

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

##### Softmax

The **Softmax** function converts a vector of raw scores into probabilities that sum to 1. It’s commonly used for multi-class classification problems.

$$
f(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

where \( x_i \) is the score for class \( i \), and the denominator is the sum of the exponentials of all scores.



### Feedforward Neural Network (FFN)    
We expand from the single perceptron to a feedforward neural network (FNN) with multiple layers. This FNN consists of an input layer, one or more hidden layers, and an output layer. Each neuron in one layer is connected to every neuron in the next layer, and information flows in one direction without loops. This type of network can handle more complex patterns and tasks compared to a single perceptron.I've written a simple code thst represents a FFN with one layer.

### Recurrent Neural Network (RRN)    

TODO

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 

## Languages and Tools
<p align="left"> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> </p>

## Requirements
```
matplotlib==3.6.3
numpy==1.24.2
TODO
```

## Test Coverage
TODO

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

<hr>

## Connect with me
<p align="left">
<a href="https://www.linkedin.com/in/francescopl/" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="francescopaololezza" height="20" width="30" /></a>
<a href="https://www.kaggle.com/francescopaolol" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/kaggle.svg" alt="francescopaololezza" height="20" width="30" /></a>
</p>



