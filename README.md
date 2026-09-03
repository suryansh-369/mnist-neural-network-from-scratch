MNIST Neural Network from Scratch

A fully connected neural network built from scratch using NumPy to classify handwritten digits from the MNIST dataset.

The goal of this project was to understand the core mechanics of a neural network by implementing the forward pass, backpropagation, parameter updates, and evaluation manually instead of relying on a high-level deep-learning framework.

Results

Baseline model

Architecture: 784 → 128 → 10

Metric

Result

Validation Accuracy

90.72%

Test Accuracy

90.85%

Architecture experiment

I also increased the hidden layer from 128 to 256 neurons:

Architecture: 784 → 256 → 10
Learning rate: 1.0
Iterations: 500

Training Accuracy: 92.59%

Validation and test accuracy should be used to determine whether this larger model is actually better at generalizing.

What I Built

MNIST data loading and preprocessing

28 × 28 image flattening into 784 input features

Pixel normalization to [0, 1]

Train/validation split

He-style weight initialization

ReLU activation for the hidden layer

Softmax output layer for 10-class classification

One-hot encoding

Manual backpropagation

Manual gradient descent

Training, validation, and test evaluation

Hyperparameter experiments

Model

The baseline network is:

28 × 28 image
     ↓
  784 inputs
     ↓
  128 neurons
     ↓
    ReLU
     ↓
  10 outputs
     ↓
   Softmax
     ↓
  Digit 0–9

The forward pass is:

Z1 = W1X + b1
A1 = ReLU(Z1)

Z2 = W2A1 + b2
A2 = Softmax(Z2)

The gradients are calculated manually during backpropagation and the parameters are updated using gradient descent.

Key Learnings

Neural network fundamentals

I learned how matrix dimensions flow through a network and why every layer must have compatible input/output shapes.

For the baseline model:

X  = (784, 50000)
W1 = (128, 784)
b1 = (128, 1)

A1 = (128, 50000)

W2 = (10, 128)
b2 = (10, 1)

A2 = (10, 50000)

Backpropagation

Instead of calling an automatic differentiation system, I calculated the gradients myself and propagated the error backward through the network.

Activation functions

I learned why ReLU is commonly used in hidden layers and why Softmax is appropriate for a single-label, multiclass problem such as MNIST.

I also explored the idea of using different activation functions in different hidden layers and how sigmoid can introduce smaller gradients.

Initialization

I learned why weights should be initialized with randomness to break symmetry, while biases can start at zero.

For ReLU layers, I used He-style initialization:

np.sqrt(2 / n_in)

Generalization

I learned the difference between:

Training accuracy — performance on data used to fit the model

Validation accuracy — used to compare and tune models

Test accuracy — final evaluation on unseen data

The baseline model achieved very similar validation and test accuracy, which was a useful indicator of reasonable generalization.

Experiments

I experimented with learning rate, training iterations, and hidden-layer size.

Experiment

Architecture

Learning Rate

Iterations

Training Accuracy

1

784 → 128 → 10

0.2

200

91.47%

2

784 → 128 → 10

0.1

500

91.47%

3

784 → 256 → 10

1.0

500

92.59%

One useful observation was that increasing the number of iterations did not automatically improve training accuracy. Changing the hidden-layer size had a larger effect in the experiments above.

Project Structure

mnist-neural-network/
├── exp.ipynb       # Experiments, training and evaluation
├── model.py        # Neural network implementation
├── README.md       # Project documentation
└── .gitignore      # Files excluded from Git

Technologies

Python

NumPy

Pandas

Matplotlib

TensorFlow/Keras dataset loader

Jupyter Notebook

Running the Project

Clone the repository and create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install numpy pandas matplotlib tensorflow

Then open exp.ipynb and run the cells in order.

Future Improvements

Add and plot cross-entropy loss during training

Visualize correct and incorrect predictions

Generate a confusion matrix

Compare additional architectures

Experiment with ReLU, sigmoid, and tanh

Implement mini-batch gradient descent

Build a PyTorch version in a separate project and train it on the NVIDIA RTX 4050 GPU

Why This Project

This project was intentionally kept NumPy + CPU.

The purpose was to understand what happens inside a neural network before moving to frameworks such as PyTorch. Implementing the mathematics manually made concepts such as matrix multiplication, activations, gradients, and parameter updates much easier to understand.

Next step: move from a NumPy implementation to a PyTorch-based model and compare the two approaches.