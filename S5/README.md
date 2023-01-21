# Assignment Submission for Session 4 - Coding Drill Down

As the assignment requires us to achieve a consistent accuracy of 99.4% in at least 3 steps and 4 steps is desirable and would make *Rohan Happy*. We have split the steps into 4 files. 

* 1st step : we set a working neural network which works
* 2nd step : we take the first stab at reducing the number of parameters
* 3rd step : we address the Overfitting aspect of the neural network
* 4th step : we meet our target of consistent 99.4% accuracy in less than 15 epochs

We learn the use of Data Augmentation techniques like Rotation, use StepLR for the model to learn faster and with lesser parameters.

## Summary of Step 1

In the [Step1 File](https://github.com/shariqfarhan/Explore/blob/master/S5/Assignment_4_Step_1.ipynb), we first setup the basic skeleton which would be the first step in building and running our model.

### Target

1.   Get the basis working neural network for MNIST up and running
2.   Import Data & Do Data processing / Transformations
3.   Set Data Loader
4.   Set Basic Working Code
5.   Set Basic Training  & Test Loop

### Result

*   Total Model Parameters :  6379786 i.e. ~6.4M
*   Best Train Accuracy :  100.0
*   Best Test Accuracy :  99.36
*   Number of Epochs with test accuracy above 99.4% threshold: 0


### Analysis

1.   Successfully created a working model for MNIST
2.   The Test accuracy never crossed the required threshold of 99.4%
3.   Model hit the highest train accuracy of 100%, that means there's some element of overfitting in the training phase
4.   The # of parameters are very high, much higher than the requirement of 8K parameters
5.   The last couple convolution layers are responsible for the explosion of number of parameters at the end


## Summary of Step 2

This [Step2 file](https://github.com/shariqfarhan/Explore/blob/master/S5/Assignment_4_Step_2.ipynb) is a continuation of the [Step1 File](https://github.com/shariqfarhan/Explore/blob/master/S5/Assignment_4_Step_1.ipynb), in the first file we setup the basic skeleton of the neural network.
Now, we want to get closer to our target.

First, we try to reduce the number of parameters. In order to do that, we need to structure the neural network in such a way that the model is learning more lesser parameters.

In the next file, we will try to achieve a consistent accuracy of more than 99.4% or higher with the parameters reduction we get in this file.


**Target**


1.   Reduce the number of parameters and get it to 10K-20K range


**Result**

*   Total Model Parameters :  12584 i.e. ~12.6K
*   Best Train Accuracy :  100.0
*   Best Test Accuracy :  98.63
*   Number of Epochs with test accuracy above 99.4% threshold: 0


**Analysis**


1.   Cut down the parameters from 6.4M to 12.6K
2.   The Test accuracy never crossed the required threshold of 99.4%
3.   Model hit the highest train accuracy of 100%, that means there's some element of overfitting in the training phase
4.   By changing the convolution layer channels we were able to reduce the number of parameters and bought it down to 12.6K but it impacted our test accuracy which never reached 99%
5.   Too many layers were involved in this round, maybe we should cut down on the number of layers and include the padding element which we had in the first stage
6.   We never addressed the overfitting aspect discusssed in the first file, that needs to be catered to

## Summary of Step 3

This [Step3 file](https://github.com/shariqfarhan/Explore/blob/master/S5/Assignment_4_Step_3.ipynb) is a continuation of the previous 2 files. 


*   In the first file we setup the basic skeleton of the neural network
*   In the second file we reduced the number of parameters from ~6.4M to ~12.6K


Now, we want to address the overfitting aspect of the neural network and improve our test accuracy.


**Target**


1.   Address the Overfitting aspect of the neural network
2.   Reduce the number of parameters and get it to 8K-10K range
3.   Consistently hit the target test accuracy of 99.4%


**Result**

* Total Model Parameters :  11924
* Best Train Accuracy :  100.0
* Best Test Accuracy :  99.43
* Num of Epochs with test accuracy above 99.4% threshold: 3





**Analysis**

1.   First, we addressed the overfitting issue by adding Batch Normalisation and adding Dropout values
2.   The Test accuracy crossed the required threshold of 99.4% - thrice and that too before 15 epochs
3.   By editing the neural network, we hit our target of 99.4% multiple times but the number of parameters is above 10K
4. We also experimented with data augmentation by adding rotation to the train dataset

## Summary of Step 4

This [Step4 file](https://github.com/shariqfarhan/Explore/blob/master/S5/Assignment_4_Step_4.ipynb) is a continuation of the previous 3 files. 

We need to hit the consistent accuracy of 99.4% with lesser parameters. To do this, we leverage Step learning rate (StepLR) to make the model learn faster as it completes few epochs.


**Target**


1.   Address the Overfitting aspect of the neural network
2.   Reduce the number of parameters and get it to 8K-10K range
3.   Consistently hit the target test accuracy of 99.4%


**Result**

*   Total Model Parameters :  9,300
*   Best Train Accuracy :  100.0
*   Best Test Accuracy :  99.52
*   Number of Epochs with test accuracy above 99.4% threshold: 14


**Analysis**

1.   First, we addressed the overfitting issue by adding Batch Normalisation and simplifying the Neural network.
1.   We leveraged StepLR to improve learning rate as the model completes more epochs
2.   Leveraging StepLR helped the model get a better accuracy with lower number of parameters

The Final model which helped us get the succesful result is as below 

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
            Conv2d-9           [-1, 16, 10, 10]           2,304
      BatchNorm2d-10           [-1, 16, 10, 10]              32
             ReLU-11           [-1, 16, 10, 10]               0
          Dropout-12           [-1, 16, 10, 10]               0
           Conv2d-13           [-1, 16, 10, 10]             256
           Conv2d-14             [-1, 16, 8, 8]           2,304
      BatchNorm2d-15             [-1, 16, 8, 8]              32
             ReLU-16             [-1, 16, 8, 8]               0
          Dropout-17             [-1, 16, 8, 8]               0
           Conv2d-18             [-1, 16, 6, 6]           2,304
      BatchNorm2d-19             [-1, 16, 6, 6]              32
             ReLU-20             [-1, 16, 6, 6]               0
           Conv2d-21             [-1, 10, 4, 4]           1,440
----------------------------------------------------------------
Total params: 9,300
Trainable params: 9,300
Non-trainable params: 0
----------------------------------------------------------------
Forward/backward pass size (MB): 0.36
Params size (MB): 0.04
Estimated Total Size (MB): 0.40
----------------------------------------------------------------
```

Also, the logs are as below.

```
EPOCH: 0
Loss=0.10401470959186554 Batch_id=937 Accuracy=88.78: 100%|██████████| 938/938 [00:22<00:00, 42.26it/s]

Test set: Average loss: 0.0695, Accuracy: 9807/10000 (98.07%)

EPOCH: 1
Loss=0.02284197323024273 Batch_id=937 Accuracy=97.58: 100%|██████████| 938/938 [00:21<00:00, 42.80it/s]

Test set: Average loss: 0.0444, Accuracy: 9856/10000 (98.56%)

EPOCH: 2
Loss=0.032893504947423935 Batch_id=937 Accuracy=98.03: 100%|██████████| 938/938 [00:24<00:00, 38.71it/s]

Test set: Average loss: 0.0324, Accuracy: 9907/10000 (99.07%)

EPOCH: 3
Loss=0.043343685567379 Batch_id=937 Accuracy=98.31: 100%|██████████| 938/938 [00:22<00:00, 42.55it/s]

Test set: Average loss: 0.0263, Accuracy: 9917/10000 (99.17%)

EPOCH: 4
Loss=0.031206805258989334 Batch_id=937 Accuracy=98.51: 100%|██████████| 938/938 [00:22<00:00, 42.52it/s]

Test set: Average loss: 0.0231, Accuracy: 9933/10000 (99.33%)

EPOCH: 5
Loss=0.01033899374306202 Batch_id=937 Accuracy=98.58: 100%|██████████| 938/938 [00:22<00:00, 42.16it/s]

Test set: Average loss: 0.0227, Accuracy: 9932/10000 (99.32%)

EPOCH: 6
Loss=0.14788119494915009 Batch_id=937 Accuracy=98.83: 100%|██████████| 938/938 [00:22<00:00, 42.52it/s]

Test set: Average loss: 0.0189, Accuracy: 9947/10000 (99.47%)

EPOCH: 7
Loss=0.027754908427596092 Batch_id=937 Accuracy=98.97: 100%|██████████| 938/938 [00:23<00:00, 40.28it/s]

Test set: Average loss: 0.0188, Accuracy: 9950/10000 (99.50%)

EPOCH: 8
Loss=0.01776299998164177 Batch_id=937 Accuracy=99.00: 100%|██████████| 938/938 [00:22<00:00, 42.51it/s]

Test set: Average loss: 0.0177, Accuracy: 9950/10000 (99.50%)

EPOCH: 9
Loss=0.004315543919801712 Batch_id=937 Accuracy=99.00: 100%|██████████| 938/938 [00:22<00:00, 42.49it/s]

Test set: Average loss: 0.0167, Accuracy: 9950/10000 (99.50%)

EPOCH: 10
Loss=0.01589987426996231 Batch_id=937 Accuracy=99.01: 100%|██████████| 938/938 [00:22<00:00, 41.94it/s]

Test set: Average loss: 0.0174, Accuracy: 9952/10000 (99.52%)

EPOCH: 11
Loss=0.08307848870754242 Batch_id=937 Accuracy=99.05: 100%|██████████| 938/938 [00:22<00:00, 42.19it/s]

Test set: Average loss: 0.0164, Accuracy: 9952/10000 (99.52%)

EPOCH: 12
Loss=0.0017476874636486173 Batch_id=937 Accuracy=99.02: 100%|██████████| 938/938 [00:23<00:00, 40.45it/s]

Test set: Average loss: 0.0167, Accuracy: 9948/10000 (99.48%)

EPOCH: 13
Loss=0.0071619353257119656 Batch_id=937 Accuracy=99.00: 100%|██████████| 938/938 [00:22<00:00, 41.75it/s]

Test set: Average loss: 0.0172, Accuracy: 9949/10000 (99.49%)

EPOCH: 14
Loss=0.008591747842729092 Batch_id=937 Accuracy=99.04: 100%|██████████| 938/938 [00:22<00:00, 42.19it/s]

Test set: Average loss: 0.0165, Accuracy: 9949/10000 (99.49%)

EPOCH: 15
Loss=0.0482780821621418 Batch_id=937 Accuracy=99.07: 100%|██████████| 938/938 [00:22<00:00, 42.31it/s]

Test set: Average loss: 0.0167, Accuracy: 9950/10000 (99.50%)

EPOCH: 16
Loss=0.025122059509158134 Batch_id=937 Accuracy=99.10: 100%|██████████| 938/938 [00:22<00:00, 42.38it/s]

Test set: Average loss: 0.0165, Accuracy: 9951/10000 (99.51%)

EPOCH: 17
Loss=0.015054437331855297 Batch_id=937 Accuracy=99.08: 100%|██████████| 938/938 [00:22<00:00, 41.46it/s]

Test set: Average loss: 0.0167, Accuracy: 9951/10000 (99.51%)

EPOCH: 18
Loss=0.045779768377542496 Batch_id=937 Accuracy=99.12: 100%|██████████| 938/938 [00:22<00:00, 42.01it/s]

Test set: Average loss: 0.0166, Accuracy: 9949/10000 (99.49%)

EPOCH: 19
Loss=0.02385699562728405 Batch_id=937 Accuracy=99.06: 100%|██████████| 938/938 [00:22<00:00, 41.75it/s]

Test set: Average loss: 0.0164, Accuracy: 9949/10000 (99.49%)


```
