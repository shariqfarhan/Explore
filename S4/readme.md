# Part 1 of the Assignment - Session 3 - Backpropagation and Architectural Basics

## Introduction

In the [Attached Excel file](https://github.com/shariqfarhan/Explore/blob/master/S4/Assignment_3_backprop.xlsx), we have showcased the inner working of the backpropagation algorithm. This readme provides an overview of the outputs. 

For a given input and output neurons, and the weights given - we identify the loss and try to reduce it. This is being done using the mean squared error loss function and the Gradient Descent algorithm.

We start with random weights for the neural network and based on the overall error calculated based on the loss function, we try to optimize the weights to have a high accuracy between the outputs & the predictions.

The above process is called "learning". Because we are trying to learn the right weights which would get our predictions closer to the output.

But, as we have seen in the class, learning rates can wreak havoc in a neural network if not controlled. The first part of the assignment is to replicate the backpropagation algorithm in an excel sheet and notice the impact various learning rates have on the loss function.

![image](https://user-images.githubusercontent.com/57046534/212425156-e7f39576-b4dc-4979-9e89-0bfbf88d85d4.png)



## Impact of learning rate on learning

Learning rate = 0.1

<img width="441" alt="image" src="https://user-images.githubusercontent.com/57046534/212279999-78dfde06-280f-4800-a3fd-e7efe4f34375.png">

Learning rate = 0.2

<img width="429" alt="image" src="https://user-images.githubusercontent.com/57046534/212280090-6a656c55-a68c-40db-8cbb-ed6cc3cfeb14.png">

Learning rate = 0.5

<img width="424" alt="image" src="https://user-images.githubusercontent.com/57046534/212280353-8183af91-978f-490e-969e-6c1fd37b7ed0.png">


Learning rate = 0.8

<img width="440" alt="image" src="https://user-images.githubusercontent.com/57046534/212280392-36b1a48a-6a2c-46ae-986b-eec295b8f85f.png">


Learning rate = 1.0

<img width="441" alt="image" src="https://user-images.githubusercontent.com/57046534/212280431-4c936e17-b0df-4ee4-a4a4-f936463b2676.png">


Learning rate = 2.0

<img width="545" alt="image" src="https://user-images.githubusercontent.com/57046534/212280483-a8e4d1ec-7efe-4205-8bdd-5fb3fa457e49.png">

## Conclusions

We see that as the learning rates increase the learning happens quicker, but faster learning could be a double-edged sword as it can impact our global minima / local minima.


# Part 2 of the Assignment - Session 3 - Backpropagation and Architectural Basics

Create a Neural network on the MNIST data in such a way that it achieves 99.4% accuracy in the validation set in <20 epochs and <20K parameters. 
Additionally, we need to leverage BN, Dropout, a Fully connected layer and GAP in making this Neural network.

We were able to achieve the 99.4% accuracy with ~12K parameters in epoch 12, 13 & 15.

Please go through the attached jupyter notebook for more detaiils on the network architecture & various steps to reach the required accuracy with minimum parameters.

# Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 4, 26, 26]              36
       BatchNorm2d-2            [-1, 4, 26, 26]               8
              ReLU-3            [-1, 4, 26, 26]               0
            Conv2d-4            [-1, 8, 24, 24]             288
       BatchNorm2d-5            [-1, 8, 24, 24]              16
              ReLU-6            [-1, 8, 24, 24]               0
            Conv2d-7           [-1, 16, 24, 24]             128
         MaxPool2d-8           [-1, 16, 12, 12]               0
            Conv2d-9           [-1, 32, 10, 10]           4,608
      BatchNorm2d-10           [-1, 32, 10, 10]              64
             ReLU-11           [-1, 32, 10, 10]               0
          Dropout-12           [-1, 32, 10, 10]               0
           Conv2d-13           [-1, 16, 10, 10]             512
      BatchNorm2d-14           [-1, 16, 10, 10]              32
             ReLU-15           [-1, 16, 10, 10]               0
          Dropout-16           [-1, 16, 10, 10]               0
           Conv2d-17             [-1, 16, 8, 8]           2,304
      BatchNorm2d-18             [-1, 16, 8, 8]              32
             ReLU-19             [-1, 16, 8, 8]               0
          Dropout-20             [-1, 16, 8, 8]               0
           Conv2d-21             [-1, 16, 6, 6]           2,304
      BatchNorm2d-22             [-1, 16, 6, 6]              32
             ReLU-23             [-1, 16, 6, 6]               0
           Conv2d-24             [-1, 10, 4, 4]           1,440
      BatchNorm2d-25             [-1, 10, 4, 4]              20
             ReLU-26             [-1, 10, 4, 4]               0
        AvgPool2d-27             [-1, 10, 1, 1]               0
           Conv2d-28             [-1, 10, 1, 1]             100
================================================================
Total params: 11,924
Trainable params: 11,924
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.45
Params size (MB): 0.05
Estimated Total Size (MB): 0.50
----------------------------------------------------------------
```

# Logs

```
EPOCH: 0
Loss=0.07375553995370865 Batch_id=937 Accuracy=88.48: 100%|██████████| 938/938 [01:10<00:00, 13.37it/s]

Test set: Average loss: 0.0535, Accuracy: 9844/10000 (98.44%)

EPOCH: 1
Loss=0.027978593483567238 Batch_id=937 Accuracy=97.56: 100%|██████████| 938/938 [01:01<00:00, 15.29it/s]

Test set: Average loss: 0.0409, Accuracy: 9889/10000 (98.89%)

EPOCH: 2
Loss=0.0711149349808693 Batch_id=937 Accuracy=98.00: 100%|██████████| 938/938 [00:59<00:00, 15.85it/s]

Test set: Average loss: 0.0292, Accuracy: 9902/10000 (99.02%)

EPOCH: 3
Loss=0.04494500532746315 Batch_id=937 Accuracy=98.28: 100%|██████████| 938/938 [01:00<00:00, 15.48it/s]

Test set: Average loss: 0.0272, Accuracy: 9920/10000 (99.20%)

EPOCH: 4
Loss=0.03568989038467407 Batch_id=937 Accuracy=98.41: 100%|██████████| 938/938 [00:58<00:00, 15.96it/s]

Test set: Average loss: 0.0243, Accuracy: 9916/10000 (99.16%)

EPOCH: 5
Loss=0.06658781319856644 Batch_id=937 Accuracy=98.53: 100%|██████████| 938/938 [00:59<00:00, 15.73it/s]

Test set: Average loss: 0.0240, Accuracy: 9925/10000 (99.25%)

EPOCH: 6
Loss=0.36980631947517395 Batch_id=937 Accuracy=98.56: 100%|██████████| 938/938 [00:59<00:00, 15.72it/s]

Test set: Average loss: 0.0228, Accuracy: 9917/10000 (99.17%)

EPOCH: 7
Loss=0.07622472941875458 Batch_id=937 Accuracy=98.64: 100%|██████████| 938/938 [00:58<00:00, 15.90it/s]

Test set: Average loss: 0.0241, Accuracy: 9929/10000 (99.29%)

EPOCH: 8
Loss=0.12336332350969315 Batch_id=937 Accuracy=98.65: 100%|██████████| 938/938 [00:59<00:00, 15.84it/s]

Test set: Average loss: 0.0224, Accuracy: 9919/10000 (99.19%)

EPOCH: 9
Loss=0.34600645303726196 Batch_id=937 Accuracy=98.72: 100%|██████████| 938/938 [01:00<00:00, 15.58it/s]

Test set: Average loss: 0.0212, Accuracy: 9935/10000 (99.35%)

EPOCH: 10
Loss=0.0028733662329614162 Batch_id=937 Accuracy=98.79: 100%|██████████| 938/938 [00:58<00:00, 16.13it/s]

Test set: Average loss: 0.0190, Accuracy: 9936/10000 (99.36%)

EPOCH: 11
Loss=0.019304955378174782 Batch_id=937 Accuracy=98.81: 100%|██████████| 938/938 [00:59<00:00, 15.73it/s]

Test set: Average loss: 0.0217, Accuracy: 9928/10000 (99.28%)

EPOCH: 12
Loss=0.01410830207169056 Batch_id=937 Accuracy=98.86: 100%|██████████| 938/938 [00:58<00:00, 15.91it/s]

Test set: Average loss: 0.0184, Accuracy: 9943/10000 (99.43%)

EPOCH: 13
Loss=0.08048190176486969 Batch_id=937 Accuracy=98.90: 100%|██████████| 938/938 [00:58<00:00, 15.94it/s]

Test set: Average loss: 0.0174, Accuracy: 9941/10000 (99.41%)

EPOCH: 14
Loss=0.018597794696688652 Batch_id=937 Accuracy=98.93: 100%|██████████| 938/938 [00:59<00:00, 15.63it/s]

Test set: Average loss: 0.0178, Accuracy: 9938/10000 (99.38%)

EPOCH: 15
Loss=0.12172360718250275 Batch_id=937 Accuracy=98.97: 100%|██████████| 938/938 [00:59<00:00, 15.75it/s]

Test set: Average loss: 0.0171, Accuracy: 9941/10000 (99.41%)

EPOCH: 16
Loss=0.019666878506541252 Batch_id=937 Accuracy=98.97: 100%|██████████| 938/938 [00:59<00:00, 15.66it/s]

Test set: Average loss: 0.0205, Accuracy: 9937/10000 (99.37%)

EPOCH: 17
Loss=0.02079240418970585 Batch_id=937 Accuracy=98.99: 100%|██████████| 938/938 [01:00<00:00, 15.44it/s]

Test set: Average loss: 0.0203, Accuracy: 9934/10000 (99.34%)

EPOCH: 18
Loss=0.045451074838638306 Batch_id=937 Accuracy=99.03: 100%|██████████| 938/938 [01:00<00:00, 15.49it/s]

Test set: Average loss: 0.0209, Accuracy: 9931/10000 (99.31%)

EPOCH: 19
Loss=0.1815895140171051 Batch_id=937 Accuracy=99.05: 100%|██████████| 938/938 [01:01<00:00, 15.34it/s]

Test set: Average loss: 0.0185, Accuracy: 9938/10000 (99.38%)

```
