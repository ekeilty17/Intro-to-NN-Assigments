# Classify an X

I created a toy classification problem, that I will use in a number of lessons in order to illustrate many core concepts in neural networks. This toy problem is as follows:

Suppose I have a grid, for now it will be a 3x3 grid, and I wish to classify this pattern in the grid

<table> 
  <tr>
    <td>1</td> <td>0</td> <td>1</td>
  </tr>
  <tr>
    <td>0</td> <td>1</td> <td>0</td>
  </tr>
  <tr>
    <td>1</td> <td>0</td> <td>1</td>
  </tr>
</table>

I will refer to this pattern as an "X" (since the 1's form the shape of an X)

## Task
We will use a Single Neuron Linear Classifier in order to solve this classification problem, i.e. a network with 9 inputs nodes to 1 output node with an activation function.

### Part 1
Can you come up with a hand-crafted solution? What should { w1, w2, ..., w9 } and the _bias_ be so that the Linear Classifier will always correctly classify the “X”?

**Extra Credit**: Does this solution scale? i.e. what if I made an “X” with a 5x5 grid instead? What if I were to pick a different arrangement of 1’s and 0’s in any grid? Can you find a hand-crafted solution to that?


### Part 2
Code a Single Neuron Neural Network that will learn to classify an “X” using gradient descent, i.e.
```math
w_i <-- w_i - (learning rate) * dE/dw_i
b <-- b - (learning rate) * dE/db
```
see L1 slides for more details.

### Part 3 (Optional)
Implement the Least Squares analytical solution to this problem. See L2 slides for more details.
