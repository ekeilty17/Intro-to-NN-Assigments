# Logistic and Multiclass Regression

The goal of this assignment is to see how binary classification and miltclass classification are related. We will do this by first analyzing some theoretical relationships and then implementing both architectures our good old classify and "X" problem,

## Task

### Part 1
Consider a Logistic Binary Classification and a Multiclass Classification with K = 2. See L3 slides for a diagram of the architecture. Find the relationship between the weights in the Logistic Classifier (**w**) and the set of weights in the Multiclass Classifier (**w1**, **w2**).

**Hint**: y_hat in the Logistic Binary Classifier and y2_hat in the Multiclass Classifier are equivalent.

### Part 2
You will derive the backpropagation formula for the Softmax activation function and Cross Entropy Loss function. This will be similar to the derivation for the sigmoid activation function, but a bit more complex. See L3 slides for more detail and some hints, it's much too mathy to write without Latex.

### Part 3
We will be solving the classify an "X" toy problem again, using the exact same deteset as before. This time you will be implementing the two architectures you analyzed in Part 1. The goal of this assignment is to get more practice with PyTorch and to get some experience with the different functions you will be using most often in practice.
