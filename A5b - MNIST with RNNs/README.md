# The MNIST Dataset
Get excited because we are done with the classifying X's and moving on to some real world problems. The [MNIST](http://yann.lecun.com/exdb/mnist/) (Modified National Institute of Standards and Technology) Dataset is a very famous dataset in the Neural Network world. It contains 70,000 (28x28) images of hand-written digits (60,000 training examples and 10,000 validation examples) ranging from 0 to 9. This dataset is often used to measure how powerful different models are, as shown in the figure below

![](https://lh3.googleusercontent.com/proxy/CHj-pxSXgM2KZB3DmWdCU3uWya8myrSmrtAyDfAO6Ndnd6ngvJ4qzRPondD3Fz7YHU3pebBCiGMTAZNwT0jtOPZSM5LO5pusNIRq)

Our goal is to use the Convolutional Neural Network architecture in order to accurately and efficiently classify these hand-written digits.

## Task

### Part 1
You will implement the LeNet architecture in PyTorch. A diagram is given below, and my own version of the diagram is provided in the L4 slides.

![LeNet](https://miro.medium.com/fit/c/1838/551/0*H9_eGAtkQXJXtkoK)

LeNet was introduced by LeCun in his paper [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) and was specifically designed for this dataset. LeNet was one of the earliest applications of the Convolutional Neural Network and marked a pivotal point in deep learning, showing the power of the idea of convolution. This very simple, compact achitecture can achive up to 96% accuracy on the MNIST dataset in a very short amount of time.

### Part 2
Try to implement an MLP architecture that can classify hand-written digits. Attempt to get over 80% accuracy. Compare the number of parameters in each model. Use the `demo.py` file to test each architecture and see failure modes of each model.
