# Explore

This Repository includes all the codes for EVA 8

In the Assignment 2.5, we use the base MNIST data and add additional inputs and use it in the output.

As we have a generic integer to be added in the input data, we add it in the NN after the processing of the image is completed.

In this way, we ensure the NN extracts all the required info from the image and this data is an addition to the image data.

# Input Data

## Image Data - x1

A 28*28 image with 1 channel

## Random Input - x2

We use random number generation to generate data. torch.randint is the method used to generate data.
We add random input to the neural network, which is being sent as one-hot encoded matrix.

## Image Output - y1

This data comes in from the MNIST data.

## Output 2 - y2

This is the sum of x2 and y1

# Data Loading

The data is loaded simultaneously to align all the variables - x1, x2, y1, y2

# Network Design

We design a neural network which first convolves over the image and the random input x2 is added after the image is convolved.
The Loss functions for output 1 was Negative log likelihood and for output 2 was a mean squared error.

# Accuracy
After 3 Epochs the test accuracy is at only 5%, perhaps this improves as we increase the number of epochs

