# Classify an X using a CNN
We are back once again to our toy problem from assignments 1 and 3. This time we are changing it up. We are still trying to classify the same 3x3 X pattern, except now it can be in a larger grid. In this assignment we will be using a 5x5 grid, but we could have used a grid of any size.

For example, these are both positive examples of an X pattern appearing in a 5x5 grid.

<table> 
  <tr>
    <td><strong>1</strong></td> <td><strong>0</strong></td> <td><strong>1</strong></td> <td>0</td> <td>0</td>
  </tr>
  <tr>
    <td><strong>0</strong></td> <td><strong>1</strong></td> <td><strong>0</strong></td> <td>0</td> <td>0</td>
  </tr>
  <tr>
    <td><strong>1</strong></td> <td><strong>0</strong></td> <td><strong>1</strong></td> <td>0</td> <td>0</td>
  </tr>
  <tr>
    <td>0</td> <td>0</td> <td>0</td> <td>0</td> <td>0</td>
  </tr>
  <tr>
    <td>0</td> <td>0</td> <td>0</td> <td>0</td> <td>0</td>
  </tr>
</table>

<table> 
  <tr>
    <td>1</td> <td>1</td> <td>1</td> <td>1</td> <td>1</td>
  </tr>
  <tr>
    <td>1</td> <td>1</td> <td><strong>1</strong></td> <td><strong>0</strong></td> <td><strong>1</strong></td> 
  </tr>
  <tr>
    <td>1</td> <td>1</td> <td><strong>0</strong></td> <td><strong>1</strong></td> <td><strong>0</strong></td>
  </tr>
  <tr>
    <td>1</td> <td>1</td> <td><strong>1</strong></td> <td><strong>0</strong></td> <td><strong>1</strong></td> 
  </tr>
  <tr>
    <td>1</td> <td>1</td> <td>1</td> <td>1</td> <td>1</td>
  </tr>
</table>

Now, our classifier needs to be translation invariant, since the X pattern can be anywhere in the grid. This is exactly what CNNs were designed for.

This is a very simplified version of image classification. In real world applications, the kernels of the CNN would learn to detect different X patterns relevant to the thing it's trying to classify. For example, if it's trying to detect human faces, one kernel may learn to locate eyes, another to locate noses, mouths, ears, etc. There are methods such as Gradient Ascent that attempt to visualize these convolutional layers, which can show the different features the kernels pick out.

## Task
You will write a ...
