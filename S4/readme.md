# Part 1 of the Assignment - Session 3 - Backpropagation and Architectural Basics

## Introduction

In the Attached Excel file, we have showcased the inner working of the backpropagation algorithm. This readme provides an overview of the outputs. 

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

We were able to achieve the 99.4% accuracy with ~11K parameters in epoch 13, 18 & 19.

Please go through the attached jupyter notebook for more detaiils on the network architecture & various steps to reach the required accuracy with minimum parameters.

# Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 4, 24, 24]             104
              ReLU-2            [-1, 4, 24, 24]               0
       BatchNorm2d-3            [-1, 4, 24, 24]               8
            Conv2d-4            [-1, 8, 22, 22]             296
              ReLU-5            [-1, 8, 22, 22]               0
       BatchNorm2d-6            [-1, 8, 22, 22]              16
         MaxPool2d-7            [-1, 8, 11, 11]               0
           Dropout-8            [-1, 8, 11, 11]               0
            Conv2d-9             [-1, 16, 9, 9]           1,168
             ReLU-10             [-1, 16, 9, 9]               0
      BatchNorm2d-11             [-1, 16, 9, 9]              32
          Dropout-12             [-1, 16, 9, 9]               0
           Conv2d-13             [-1, 32, 7, 7]           4,640
             ReLU-14             [-1, 32, 7, 7]               0
      BatchNorm2d-15             [-1, 32, 7, 7]              64
           Conv2d-16             [-1, 16, 5, 5]           4,624
             ReLU-17             [-1, 16, 5, 5]               0
      BatchNorm2d-18             [-1, 16, 5, 5]              32
          Dropout-19             [-1, 16, 5, 5]               0
        AvgPool2d-20             [-1, 16, 1, 1]               0
           Linear-21                   [-1, 10]             170
================================================================
Total params: 11,154
Trainable params: 11,154
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.24
Params size (MB): 0.04
Estimated Total Size (MB): 0.29
----------------------------------------------------------------
```

# Logs

```

loss=0.18145139515399933 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.55it/s]Epoch : 1


Test set: Average loss: 0.0894, Accuracy: 9754/10000 (97.54%)

loss=0.04759661480784416 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.77it/s]Epoch : 2


Test set: Average loss: 0.0465, Accuracy: 9859/10000 (98.59%)

loss=0.05069602653384209 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.88it/s]Epoch : 3


Test set: Average loss: 0.0405, Accuracy: 9868/10000 (98.68%)

loss=0.07508888840675354 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.35it/s]Epoch : 4


Test set: Average loss: 0.0319, Accuracy: 9900/10000 (99.00%)

loss=0.02552906610071659 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.91it/s]Epoch : 5


Test set: Average loss: 0.0347, Accuracy: 9894/10000 (98.94%)

loss=0.09159165620803833 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 29.71it/s]Epoch : 6


Test set: Average loss: 0.0351, Accuracy: 9896/10000 (98.96%)

loss=0.09653136879205704 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 31.04it/s]Epoch : 7


Test set: Average loss: 0.0262, Accuracy: 9920/10000 (99.20%)

loss=0.063936248421669 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.18it/s]Epoch : 8


Test set: Average loss: 0.0252, Accuracy: 9922/10000 (99.22%)

loss=0.03218716010451317 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.58it/s]Epoch : 9


Test set: Average loss: 0.0242, Accuracy: 9925/10000 (99.25%)

loss=0.031701892614364624 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.78it/s]Epoch : 10


Test set: Average loss: 0.0243, Accuracy: 9934/10000 (99.34%)

loss=0.01010225247591734 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 33.14it/s]Epoch : 11


Test set: Average loss: 0.0221, Accuracy: 9930/10000 (99.30%)

loss=0.020196281373500824 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.80it/s]Epoch : 12


Test set: Average loss: 0.0263, Accuracy: 9923/10000 (99.23%)

loss=0.023443201556801796 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 33.29it/s]Epoch : 13


Test set: Average loss: 0.0213, Accuracy: 9940/10000 (99.40%)

loss=0.055004268884658813 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.61it/s]Epoch : 14


Test set: Average loss: 0.0221, Accuracy: 9935/10000 (99.35%)

loss=0.010417886078357697 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 31.14it/s]Epoch : 15


Test set: Average loss: 0.0221, Accuracy: 9934/10000 (99.34%)

loss=0.015014395117759705 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 33.09it/s]Epoch : 16


Test set: Average loss: 0.0208, Accuracy: 9936/10000 (99.36%)

loss=0.006766256410628557 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.36it/s]Epoch : 17


Test set: Average loss: 0.0208, Accuracy: 9939/10000 (99.39%)

loss=0.05433541536331177 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.15it/s]Epoch : 18


Test set: Average loss: 0.0191, Accuracy: 9940/10000 (99.40%)

loss=0.04674798622727394 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.93it/s]Epoch : 19


Test set: Average loss: 0.0207, Accuracy: 9942/10000 (99.42%)

loss=0.006371767725795507 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 33.05it/s]Epoch : 20


Test set: Average loss: 0.0211, Accuracy: 9936/10000 (99.36%)

```
