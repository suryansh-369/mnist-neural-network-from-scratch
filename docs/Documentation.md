MNIST Neural Network from Scratch using NumPy

A neural network built from scratch with NumPy to classify handwritten digits from the MNIST dataset.

The main goal of this project was not just to get a high accuracy, but to understand what is happening inside a neural network by implementing the important pieces manually:

Forward propagation

ReLU activation

Softmax activation

One-hot encoding

Backpropagation

Gradient descent

Weight initialization

Training/validation splitting

Accuracy evaluation

Experimenting with learning rate and hidden-layer size

Important: This project intentionally uses NumPy for the neural-network calculations and runs on the CPU. PyTorch/CUDA was set up separately for future projects, but is not used to train this model.

Project Overview

The network takes a flattened 28 × 28 MNIST image:

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

A second experiment used a larger hidden layer:

784 → 256 → 10

Dataset

The MNIST dataset was loaded using:

from tensorflow.keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

The original dataset contains:

Training images: 60,000
Test images:     10,000
Image size:      28 × 28
Classes:         10 (digits 0–9)

Train / Validation Split

The original 60,000 training examples were split into:

Training:   50,000
Validation: 10,000
Test:       10,000

The validation set was taken from the beginning of the original training set:

X_val = X_train_full[:10000]
Y_val = Y_train_full[:10000]

X_train = X_train_full[10000:]
Y_train = Y_train_full[10000:]

The test set was kept separate for final evaluation.

Data Preprocessing

1. Flattening

Each 28 × 28 image was converted into a vector of 784 pixels:

X_train = X_train.reshape(X_train.shape[0], 784)

2. Transposing

The data was transposed so that each image became a column:

X_train.shape = (784, 50000)
X_val.shape   = (784, 10000)

This matches the matrix equations used in the neural network.

3. Normalization

Pixel values were changed from:

0 → 255

to:

0 → 1

using:

X_train = X_train / 255.0
X_val = X_val / 255.0

Neural Network Implementation

1. Weight Initialization

The network uses randomized weights so that neurons do not start with identical parameters.

For ReLU layers, He-style initialization was used:

W1 = np.random.randn(128, 784) * np.sqrt(2 / 784)
W2 = np.random.randn(10, 128) * np.sqrt(2 / 128)

Biases were initialized to zero:

b1 = np.zeros((128, 1))
b2 = np.zeros((10, 1))

Why random weights?

If all weights started at zero, neurons in the same layer would learn the same thing because they would have identical gradients.

Random weights break this symmetry.

Why zero biases?

Biases do not need random initialization to break symmetry because the weights are already different. Starting biases at zero provides a neutral starting point and lets gradient descent learn the required bias values.

2. Forward Propagation

The first layer calculates:

Z1 = W1.dot(X) + b1
A1 = relu(Z1)

The output layer calculates:

Z2 = W2.dot(A1) + b2
A2 = softmax(Z2)

So mathematically:

Z1 = W1X + b1
A1 = ReLU(Z1)

Z2 = W2A1 + b2
A2 = Softmax(Z2)

3. ReLU

The ReLU function is:

def relu(Z):
    return np.maximum(0, Z)

Mathematically:

ReLU(x) = max(0, x)

ReLU was used in the hidden layer because it is simple and generally provides better gradient flow than sigmoid for this type of network.

4. Softmax

Softmax converts the final output scores into values that form a probability distribution:

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)

The subtraction of the maximum value is a numerical-stability trick.

For MNIST, there are 10 mutually exclusive classes, so softmax is a natural output activation.

Example:

0 → 0.01
1 → 0.02
2 → 0.80
3 → 0.03
...

The largest probability becomes the predicted digit.

5. One-Hot Encoding

The labels are converted into one-hot vectors for the output layer.

For example:

digit 3

becomes approximately:

[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

6. Backpropagation

The gradients were calculated manually.

For the output layer:

dz2 = A2 - one_hot_y
dw2 = 1/m * dz2.dot(A1.T)
db2 = 1/m * np.sum(dz2, axis=1, keepdims=True)

Then the error was propagated back into the hidden layer:

dz1 = W2.T.dot(dz2) * deriv_relu(Z1)
dw1 = 1/m * dz1.dot(X.T)
db1 = 1/m * np.sum(dz1, axis=1, keepdims=True)

This was one of the main learning goals of the project: understanding how the error signal travels backward through the network.

7. Gradient Descent

The parameters are updated manually:

W1 = W1 - alpha * dw1
b1 = b1 - alpha * db1

W2 = W2 - alpha * dw2
b2 = b2 - alpha * db2

where alpha is the learning rate.

Results

Initial Model

Architecture:

784 → 128 → 10

One experiment used:

Learning rate: 0.2
Iterations:    200

Training accuracy at the end of that run:

91.474%

Validation accuracy:

90.72%

Test accuracy:

90.85%

The small difference between validation and test accuracy was a positive sign that the model was generalizing reasonably well.

Experiment 2

Architecture:

784 → 128 → 10

Settings:

Learning rate: 0.1
Iterations:    500

Final training accuracy:

91.474%

This showed that simply increasing the number of iterations did not automatically improve the final training accuracy for this particular setup.

Experiment 3

Architecture:

784 → 256 → 10

Settings:

Learning rate: 1.0
Iterations:    500

Final training accuracy:

92.59%

This experiment suggested that increasing the hidden-layer size gave the model more capacity to learn.

The validation and test performance for this experiment should be evaluated before declaring it the best model.

What I Learned

This project was mainly a learning exercise, and it helped me understand several concepts that are often hidden behind high-level deep-learning libraries.

Neural network fundamentals

I learned how a neural network transforms an input through multiple layers:

Input → Linear transformation → Activation → Output

I also learned how matrix dimensions must line up across layers.

For the 784 → 128 → 10 model:

X  = (784, 50000)
W1 = (128, 784)
b1 = (128, 1)

A1 = (128, 50000)

W2 = (10, 128)
b2 = (10, 1)

A2 = (10, 50000)

Understanding these shapes made matrix multiplication and the forward pass much clearer.

Forward propagation vs backpropagation

I learned that:

Forward propagation calculates the prediction.

Input
 ↓
Hidden layer
 ↓
Output

Backpropagation calculates how each parameter contributed to the error.

Output error
 ↓
Output layer gradients
 ↓
Hidden layer gradients
 ↓
Parameter updates

Activation functions

I learned that different activation functions have different purposes.

ReLU

Useful for hidden layers:

ReLU(x) = max(0, x)

Sigmoid

Produces values between 0 and 1, but its derivatives can become very small, which can lead to vanishing gradients.

Softmax

Useful for multiclass classification where exactly one class should be selected.

I also learned that softmax is not required for every classification problem:

Binary classification often uses sigmoid.

Multilabel classification often uses independent sigmoids.

Multiclass single-label classification commonly uses softmax.

Why biases can start at zero

I learned that weights need random initialization to break symmetry, but biases can start at zero because the random weights already make neurons different.

Training accuracy vs validation accuracy vs test accuracy

I learned that:

Training accuracy tells how well the model fits the training data.

Validation accuracy helps compare models and hyperparameters.

Test accuracy should be used as a final evaluation after model decisions are made.

A high training accuracy alone does not prove that a model generalizes well.

Hyperparameters matter

I experimented with:

Learning rate

Number of iterations

Number of hidden neurons

The experiments showed that changing the architecture can affect performance, while simply increasing the number of iterations does not necessarily guarantee better accuracy.

CPU vs GPU

I learned that NumPy operations run on the CPU in this project.

I also set up a CUDA-enabled PyTorch environment and confirmed that an NVIDIA RTX 4050 Laptop GPU was available.

However, this project intentionally remains NumPy + CPU so that the implementation stays focused on understanding the neural-network mathematics.

PyTorch + GPU/CUDA will be used in a future project.

Next Improvements

The following are natural next steps for this project:

Loss calculation

Add cross-entropy loss and track:

Loss ↓
Accuracy ↑

during training.

Prediction visualization

Display handwritten digits with:

Predicted label
Actual label

to inspect individual predictions.

Confusion matrix

Analyze which digits are commonly confused, such as:

5 → 3
4 → 9

More architectures

Experiment with:

784 → 64 → 10
784 → 128 → 10
784 → 256 → 10
784 → 256 → 128 → 10

Different activations

Compare:

ReLU → ReLU
ReLU → Sigmoid
ReLU → Tanh

and observe how activation choice affects learning.

Mini-batch gradient descent

The current implementation uses full-batch gradient descent. A future improvement is to train using smaller batches such as:

64
128
256

This would make the training approach closer to how practical neural networks are usually trained.

Project Structure

A simple version of the project can be organized as:

neural_network/
│
├── exp.ipynb
├── model.py
├── README.md
└── .gitignore

Installation

Create and activate a virtual environment, then install the dependencies:

pip install numpy pandas matplotlib tensorflow

Run the notebook with the project's Python environment selected as the Jupyter kernel.

Running the Project

Open:

exp.ipynb

and run the cells in order:

1. Load MNIST
2. Split train / validation data
3. Preprocess the data
4. Define the neural-network functions
5. Train with gradient descent
6. Evaluate on validation data
7. Evaluate on test data
8. Run experiments and visualizations

Why I Built This

This project was intentionally built from the mathematics upward instead of starting with a high-level framework.

The goal was to understand:

What actually happens inside a neural network?

Before moving on to frameworks such as PyTorch, I wanted to be comfortable implementing the core operations myself.

Future Direction

The next project will move from:

NumPy + manual neural network + CPU

to:

PyTorch + GPU/CUDA + more practical deep learning

The NVIDIA RTX 4050 Laptop GPU is already configured and available for that next stage.