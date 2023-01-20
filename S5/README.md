# Assignment Submission for Session 4 - Coding Drill Down

As the assignment requires us to achieve a consistent accuracy of 99.4% in at least 3 steps and 4 steps is desirable and would make *Rohan Happy*. We have split the steps into 4 files. 

* 1st step : we set a working neural network which works
* 2nd step : we take the first stab at reducing the number of parameters
* 3rd step : we address the Overfitting aspect of the neural network
* 4th step : we meet our target of consistent 99.4% accuracy in less than 15 epochs

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

Here is the model structure for this model.

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================

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

Here are the logs for this model.

EPOCH: 0
Loss=0.1817273050546646 Batch_id=937 Accuracy=89.99: 100%|██████████| 938/938 [01:04<00:00, 14.58it/s]

Test set: Average loss: 0.0469, Accuracy: 9860/10000 (98.60%)

EPOCH: 1
Loss=0.024927718564867973 Batch_id=937 Accuracy=97.95: 100%|██████████| 938/938 [00:54<00:00, 17.25it/s]

Test set: Average loss: 0.0369, Accuracy: 9891/10000 (98.91%)

EPOCH: 2
Loss=0.02568667009472847 Batch_id=937 Accuracy=98.33: 100%|██████████| 938/938 [00:53<00:00, 17.45it/s]

Test set: Average loss: 0.0292, Accuracy: 9908/10000 (99.08%)

EPOCH: 3
Loss=0.020776044577360153 Batch_id=937 Accuracy=98.57: 100%|██████████| 938/938 [00:54<00:00, 17.30it/s]

Test set: Average loss: 0.0294, Accuracy: 9910/10000 (99.10%)

EPOCH: 4
Loss=0.0651998221874237 Batch_id=937 Accuracy=98.73: 100%|██████████| 938/938 [00:55<00:00, 16.78it/s]

Test set: Average loss: 0.0252, Accuracy: 9915/10000 (99.15%)

EPOCH: 5
Loss=0.019690189510583878 Batch_id=937 Accuracy=98.77: 100%|██████████| 938/938 [00:56<00:00, 16.68it/s]

Test set: Average loss: 0.0226, Accuracy: 9931/10000 (99.31%)

EPOCH: 6
Loss=0.006541811861097813 Batch_id=937 Accuracy=98.87: 100%|██████████| 938/938 [00:55<00:00, 16.85it/s]

Test set: Average loss: 0.0213, Accuracy: 9933/10000 (99.33%)

EPOCH: 7
Loss=0.046682123094797134 Batch_id=937 Accuracy=98.88: 100%|██████████| 938/938 [00:55<00:00, 16.81it/s]

Test set: Average loss: 0.0222, Accuracy: 9930/10000 (99.30%)

EPOCH: 8
Loss=0.0017001306405290961 Batch_id=937 Accuracy=99.00: 100%|██████████| 938/938 [00:57<00:00, 16.32it/s]

Test set: Average loss: 0.0196, Accuracy: 9936/10000 (99.36%)

EPOCH: 9
Loss=0.11725009977817535 Batch_id=937 Accuracy=98.99: 100%|██████████| 938/938 [00:55<00:00, 16.99it/s]

Test set: Average loss: 0.0187, Accuracy: 9945/10000 (99.45%)

EPOCH: 10
Loss=0.0038126646541059017 Batch_id=937 Accuracy=99.00: 100%|██████████| 938/938 [00:55<00:00, 16.90it/s]

Test set: Average loss: 0.0190, Accuracy: 9940/10000 (99.40%)

EPOCH: 11
Loss=0.0653543472290039 Batch_id=937 Accuracy=99.05: 100%|██████████| 938/938 [00:56<00:00, 16.59it/s]

Test set: Average loss: 0.0236, Accuracy: 9925/10000 (99.25%)

EPOCH: 12
Loss=0.13482144474983215 Batch_id=937 Accuracy=99.10: 100%|██████████| 938/938 [00:55<00:00, 17.02it/s]

Test set: Average loss: 0.0189, Accuracy: 9943/10000 (99.43%)

EPOCH: 13
Loss=0.012560486793518066 Batch_id=937 Accuracy=99.11: 100%|██████████| 938/938 [00:55<00:00, 16.77it/s]

Test set: Average loss: 0.0234, Accuracy: 9935/10000 (99.35%)

EPOCH: 14
Loss=0.014288542792201042 Batch_id=937 Accuracy=99.13: 100%|██████████| 938/938 [00:56<00:00, 16.52it/s]

Test set: Average loss: 0.0202, Accuracy: 9935/10000 (99.35%)

EPOCH: 15
Loss=0.02761080674827099 Batch_id=937 Accuracy=99.14: 100%|██████████| 938/938 [00:56<00:00, 16.55it/s]

Test set: Average loss: 0.0208, Accuracy: 9936/10000 (99.36%)

EPOCH: 16
Loss=0.00967184267938137 Batch_id=937 Accuracy=99.21: 100%|██████████| 938/938 [00:55<00:00, 16.99it/s]

Test set: Average loss: 0.0216, Accuracy: 9929/10000 (99.29%)

EPOCH: 17
Loss=0.13152864575386047 Batch_id=937 Accuracy=99.14: 100%|██████████| 938/938 [00:55<00:00, 16.81it/s]

Test set: Average loss: 0.0200, Accuracy: 9930/10000 (99.30%)

EPOCH: 18
Loss=0.014984813518822193 Batch_id=937 Accuracy=99.25: 100%|██████████| 938/938 [00:58<00:00, 15.97it/s]

Test set: Average loss: 0.0178, Accuracy: 9950/10000 (99.50%)

EPOCH: 19
Loss=0.0495365709066391 Batch_id=937 Accuracy=99.24: 100%|██████████| 938/938 [00:56<00:00, 16.57it/s]

Test set: Average loss: 0.0230, Accuracy: 9926/10000 (99.26%)
