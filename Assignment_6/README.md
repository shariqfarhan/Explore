# Assignment Submission for Session 6 - Advanced Concepts

1. Run this [network](https://colab.research.google.com/drive/1qlewMtxcAJT6fIJdmMh8pSf2e-dh51Rw) 
2. Fix the network above:
    1. change the code such that it uses GPU and
    2. change the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
    3. total RF must be more than 44
    4. one of the layers must use Depthwise Separable Convolution
    5. one of the layers must use Dilated Convolution
    6. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
    7. use albumentation library and apply:
          1. horizontal flip
          2. shiftScaleRotate
          3. coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
    8. achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.
    9. upload to Github

# Summary

1. Total Model Parameters :  151902
2. Best Train Accuracy :  88.28125
3. Best Test Accuracy :  86.45
4. Number of Epochs with test accuracy above 85% threshold: 13
5. Receptive Field - 109


# Receptive Field Calculation for the model

```

Input_features	padding	kernel	stride	dilation	jump_In	jump_out	RF_In	RF_Out	Output_Feature	Input_Channel	Output_Channels	kernel_adjusted	Convolution Type
32	1	3	1	1	1	1	1	3	32	3	16	3	Normal
32	1	3	1	1	1	1	3	5	32	16	32	3	Normal
32	1	3	1	1	1	1	5	7	32	32	64	3	Normal
32	1	3	1	1	1	1	7	9	32	64	64	3	Normal
32	1	3	2	1	1	2	7	9	16	64	64	3	Strided Conv
16	1	3	1	1	2	2	9	13	16	64	128	3	Depthwise
16	0	1	1	1	2	2	13	13	16	128	128	1	Pointwise Conv
16	1	3	1	1	2	2	13	17	16	128	256	3	Depthwise
16	0	1	1	1	2	2	17	17	16	256	256	1	Pointwise Conv
16	1	3	2	1	2	4	17	21	8	512	512	3	Strided Conv
8	1	3	1	1	4	4	21	29	8	512	256	3	Depthwise
8	0	1	1	1	4	4	29	29	8	256	128	1	Pointwise Conv
8	1	3	1	1	4	4	29	37	8	128	64	3	Depthwise
8	0	1	1	1	4	4	37	37	8	64	64	1	Pointwise Conv
8	1	3	2	1	4	8	37	45	4	64	64	3	Strided Conv
4	1	3	1	1	8	8	45	61	4	64	64	3	Depthwise
4	0	1	1	1	8	8	61	61	4	64	64	1	Pointwise Conv
4	2	3	1	2	8	8	61	93	4	64	64	5	Dilated Conv
4	1	3	1	1	8	8	93	109	4	64	64	3	Normal 

```

<img width="1061" alt="image" src="https://user-images.githubusercontent.com/57046534/218267684-a7689ca5-a2d5-4007-82ed-a7dad3f4bbbb.png">


# Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
              ReLU-2           [-1, 16, 32, 32]               0
       BatchNorm2d-3           [-1, 16, 32, 32]              32
           Dropout-4           [-1, 16, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           4,608
              ReLU-6           [-1, 32, 32, 32]               0
       BatchNorm2d-7           [-1, 32, 32, 32]              64
           Dropout-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 32, 32, 32]           9,216
             ReLU-10           [-1, 32, 32, 32]               0
      BatchNorm2d-11           [-1, 32, 32, 32]              64
          Dropout-12           [-1, 32, 32, 32]               0
           Conv2d-13           [-1, 32, 32, 32]           9,216
             ReLU-14           [-1, 32, 32, 32]               0
      BatchNorm2d-15           [-1, 32, 32, 32]              64
          Dropout-16           [-1, 32, 32, 32]               0
           Conv2d-17           [-1, 64, 16, 16]          18,432
             ReLU-18           [-1, 64, 16, 16]               0
      BatchNorm2d-19           [-1, 64, 16, 16]             128
          Dropout-20           [-1, 64, 16, 16]               0
           Conv2d-21           [-1, 64, 16, 16]             576
           Conv2d-22          [-1, 128, 16, 16]           8,192
             ReLU-23          [-1, 128, 16, 16]               0
      BatchNorm2d-24          [-1, 128, 16, 16]             256
          Dropout-25          [-1, 128, 16, 16]               0
           Conv2d-26          [-1, 128, 16, 16]           1,152
           Conv2d-27           [-1, 64, 16, 16]           8,192
             ReLU-28           [-1, 64, 16, 16]               0
      BatchNorm2d-29           [-1, 64, 16, 16]             128
          Dropout-30           [-1, 64, 16, 16]               0
           Conv2d-31             [-1, 32, 8, 8]          18,432
             ReLU-32             [-1, 32, 8, 8]               0
      BatchNorm2d-33             [-1, 32, 8, 8]              64
          Dropout-34             [-1, 32, 8, 8]               0
           Conv2d-35             [-1, 32, 8, 8]             288
           Conv2d-36             [-1, 64, 8, 8]           2,048
             ReLU-37             [-1, 64, 8, 8]               0
      BatchNorm2d-38             [-1, 64, 8, 8]             128
          Dropout-39             [-1, 64, 8, 8]               0
           Conv2d-40             [-1, 64, 8, 8]             576
           Conv2d-41             [-1, 64, 8, 8]           4,096
             ReLU-42             [-1, 64, 8, 8]               0
      BatchNorm2d-43             [-1, 64, 8, 8]             128
          Dropout-44             [-1, 64, 8, 8]               0
           Conv2d-45             [-1, 64, 4, 4]          36,864
             ReLU-46             [-1, 64, 4, 4]               0
      BatchNorm2d-47             [-1, 64, 4, 4]             128
          Dropout-48             [-1, 64, 4, 4]               0
           Conv2d-49            [-1, 128, 4, 4]           1,152
           Conv2d-50             [-1, 64, 4, 4]           8,192
             ReLU-51             [-1, 64, 4, 4]               0
      BatchNorm2d-52             [-1, 64, 4, 4]             128
          Dropout-53             [-1, 64, 4, 4]               0
           Conv2d-54             [-1, 32, 4, 4]          18,432
             ReLU-55             [-1, 32, 4, 4]               0
      BatchNorm2d-56             [-1, 32, 4, 4]              64
          Dropout-57             [-1, 32, 4, 4]               0
           Conv2d-58             [-1, 10, 4, 4]             320
        AvgPool2d-59             [-1, 10, 1, 1]               0
           Linear-60                   [-1, 10]             110
================================================================
Total params: 151,902
Trainable params: 151,902
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.33
Params size (MB): 0.58
Estimated Total Size (MB): 6.92
----------------------------------------------------------------

```

# Training Logs

Unable to train in Colab for more than 12 epochs. If we can train for few more, we can hit the target of 85% in less than 200K params.

```
EPOCH: 0
Loss=1.6539841890335083 Batch_id=781 Accuracy=34.78: 100%|██████████| 782/782 [00:21<00:00, 35.72it/s]

Test set: Average loss: 1.3916, Accuracy: 4910/10000 (49.10%)

EPOCH: 1
Loss=1.3953696489334106 Batch_id=781 Accuracy=47.74: 100%|██████████| 782/782 [00:20<00:00, 37.70it/s]

Test set: Average loss: 1.1406, Accuracy: 5883/10000 (58.83%)

EPOCH: 2
Loss=1.3590184450149536 Batch_id=781 Accuracy=53.97: 100%|██████████| 782/782 [00:20<00:00, 37.64it/s]

Test set: Average loss: 1.0282, Accuracy: 6324/10000 (63.24%)

EPOCH: 3
Loss=1.4220689535140991 Batch_id=781 Accuracy=57.64: 100%|██████████| 782/782 [00:20<00:00, 37.26it/s]

Test set: Average loss: 0.9438, Accuracy: 6622/10000 (66.22%)

EPOCH: 4
Loss=1.167194128036499 Batch_id=781 Accuracy=60.37: 100%|██████████| 782/782 [00:20<00:00, 37.45it/s]

Test set: Average loss: 0.8726, Accuracy: 6935/10000 (69.35%)

EPOCH: 5
Loss=0.6591940522193909 Batch_id=781 Accuracy=62.16: 100%|██████████| 782/782 [00:20<00:00, 37.71it/s]

Test set: Average loss: 0.7885, Accuracy: 7244/10000 (72.44%)

EPOCH: 6
Loss=1.0138815641403198 Batch_id=781 Accuracy=64.57: 100%|██████████| 782/782 [00:20<00:00, 37.53it/s]

Test set: Average loss: 0.7959, Accuracy: 7235/10000 (72.35%)

EPOCH: 7
Loss=1.331690788269043 Batch_id=781 Accuracy=65.32: 100%|██████████| 782/782 [00:21<00:00, 37.20it/s]

Test set: Average loss: 0.7649, Accuracy: 7300/10000 (73.00%)

EPOCH: 8
Loss=1.1082392930984497 Batch_id=781 Accuracy=66.93: 100%|██████████| 782/782 [00:21<00:00, 36.60it/s]

Test set: Average loss: 0.6931, Accuracy: 7577/10000 (75.77%)

EPOCH: 9
Loss=1.5621273517608643 Batch_id=781 Accuracy=67.51: 100%|██████████| 782/782 [00:20<00:00, 37.24it/s]

Test set: Average loss: 0.6895, Accuracy: 7576/10000 (75.76%)

EPOCH: 10
Loss=1.2854598760604858 Batch_id=781 Accuracy=68.55: 100%|██████████| 782/782 [00:21<00:00, 36.74it/s]

Test set: Average loss: 0.6529, Accuracy: 7736/10000 (77.36%)

EPOCH: 11
Loss=0.8199167251586914 Batch_id=781 Accuracy=69.40: 100%|██████████| 782/782 [00:21<00:00, 36.93it/s]

Test set: Average loss: 0.6416, Accuracy: 7785/10000 (77.85%)

EPOCH: 12
Loss=0.9428573846817017 Batch_id=781 Accuracy=69.96: 100%|██████████| 782/782 [00:20<00:00, 37.33it/s]

Test set: Average loss: 0.6304, Accuracy: 7829/10000 (78.29%)

EPOCH: 13
Loss=1.031165361404419 Batch_id=781 Accuracy=70.49: 100%|██████████| 782/782 [00:20<00:00, 37.33it/s]

Test set: Average loss: 0.6117, Accuracy: 7891/10000 (78.91%)

EPOCH: 14
Loss=1.174709439277649 Batch_id=781 Accuracy=71.37: 100%|██████████| 782/782 [00:21<00:00, 37.07it/s]

Test set: Average loss: 0.6140, Accuracy: 7859/10000 (78.59%)

EPOCH: 15
Loss=0.8991893529891968 Batch_id=781 Accuracy=71.94: 100%|██████████| 782/782 [00:21<00:00, 36.93it/s]

Test set: Average loss: 0.5737, Accuracy: 8070/10000 (80.70%)

EPOCH: 16
Loss=0.7198785543441772 Batch_id=781 Accuracy=72.19: 100%|██████████| 782/782 [00:21<00:00, 37.03it/s]

Test set: Average loss: 0.6065, Accuracy: 7932/10000 (79.32%)

EPOCH: 17
Loss=1.0639640092849731 Batch_id=781 Accuracy=72.80: 100%|██████████| 782/782 [00:21<00:00, 36.78it/s]

Test set: Average loss: 0.5451, Accuracy: 8113/10000 (81.13%)

EPOCH: 18
Loss=0.8722699880599976 Batch_id=781 Accuracy=73.10: 100%|██████████| 782/782 [00:20<00:00, 37.44it/s]

Test set: Average loss: 0.5503, Accuracy: 8098/10000 (80.98%)

EPOCH: 19
Loss=0.7459219098091125 Batch_id=781 Accuracy=73.42: 100%|██████████| 782/782 [00:20<00:00, 37.30it/s]

Test set: Average loss: 0.5348, Accuracy: 8159/10000 (81.59%)

EPOCH: 20
Loss=0.9422125816345215 Batch_id=781 Accuracy=73.85: 100%|██████████| 782/782 [00:21<00:00, 37.19it/s]

Test set: Average loss: 0.5300, Accuracy: 8197/10000 (81.97%)

EPOCH: 21
Loss=0.46932291984558105 Batch_id=781 Accuracy=74.07: 100%|██████████| 782/782 [00:21<00:00, 37.23it/s]

Test set: Average loss: 0.5318, Accuracy: 8188/10000 (81.88%)

EPOCH: 22
Loss=0.7251648306846619 Batch_id=781 Accuracy=74.42: 100%|██████████| 782/782 [00:21<00:00, 37.22it/s]

Test set: Average loss: 0.5189, Accuracy: 8246/10000 (82.46%)

EPOCH: 23
Loss=1.4151018857955933 Batch_id=781 Accuracy=74.75: 100%|██████████| 782/782 [00:20<00:00, 37.29it/s]

Test set: Average loss: 0.5168, Accuracy: 8223/10000 (82.23%)

EPOCH: 24
Loss=1.46941339969635 Batch_id=781 Accuracy=75.41: 100%|██████████| 782/782 [00:20<00:00, 37.27it/s]

Test set: Average loss: 0.5270, Accuracy: 8170/10000 (81.70%)

EPOCH: 25
Loss=0.6714040637016296 Batch_id=781 Accuracy=75.35: 100%|██████████| 782/782 [00:20<00:00, 37.26it/s]

Test set: Average loss: 0.4937, Accuracy: 8287/10000 (82.87%)

EPOCH: 26
Loss=0.8577061295509338 Batch_id=781 Accuracy=75.93: 100%|██████████| 782/782 [00:20<00:00, 37.27it/s]

Test set: Average loss: 0.4800, Accuracy: 8366/10000 (83.66%)

EPOCH: 27
Loss=1.054166555404663 Batch_id=781 Accuracy=75.56: 100%|██████████| 782/782 [00:21<00:00, 37.16it/s]

Test set: Average loss: 0.4968, Accuracy: 8327/10000 (83.27%)

EPOCH: 28
Loss=0.8421551585197449 Batch_id=781 Accuracy=76.09: 100%|██████████| 782/782 [00:21<00:00, 36.61it/s]

Test set: Average loss: 0.4773, Accuracy: 8364/10000 (83.64%)

EPOCH: 29
Loss=0.40768325328826904 Batch_id=781 Accuracy=76.61: 100%|██████████| 782/782 [00:21<00:00, 37.23it/s]

Test set: Average loss: 0.4726, Accuracy: 8389/10000 (83.89%)

EPOCH: 30
Loss=1.2624924182891846 Batch_id=781 Accuracy=76.59: 100%|██████████| 782/782 [00:20<00:00, 37.26it/s]

Test set: Average loss: 0.4718, Accuracy: 8424/10000 (84.24%)

EPOCH: 31
Loss=0.6790841817855835 Batch_id=781 Accuracy=76.67: 100%|██████████| 782/782 [00:21<00:00, 37.13it/s]

Test set: Average loss: 0.4758, Accuracy: 8399/10000 (83.99%)

EPOCH: 32
Loss=0.996164858341217 Batch_id=781 Accuracy=77.13: 100%|██████████| 782/782 [00:21<00:00, 37.03it/s]

Test set: Average loss: 0.4753, Accuracy: 8368/10000 (83.68%)

EPOCH: 33
Loss=0.6423828601837158 Batch_id=781 Accuracy=77.49: 100%|██████████| 782/782 [00:21<00:00, 37.01it/s]

Test set: Average loss: 0.4572, Accuracy: 8453/10000 (84.53%)

EPOCH: 34
Loss=1.3193395137786865 Batch_id=781 Accuracy=77.43: 100%|██████████| 782/782 [00:21<00:00, 37.08it/s]

Test set: Average loss: 0.4651, Accuracy: 8447/10000 (84.47%)

EPOCH: 35
Loss=0.3585742712020874 Batch_id=781 Accuracy=77.33: 100%|██████████| 782/782 [00:21<00:00, 37.16it/s]

Test set: Average loss: 0.4470, Accuracy: 8464/10000 (84.64%)

EPOCH: 36
Loss=0.502230703830719 Batch_id=781 Accuracy=77.90: 100%|██████████| 782/782 [00:21<00:00, 37.07it/s]

Test set: Average loss: 0.4398, Accuracy: 8516/10000 (85.16%)

EPOCH: 37
Loss=0.9931371212005615 Batch_id=781 Accuracy=77.66: 100%|██████████| 782/782 [00:20<00:00, 37.24it/s]

Test set: Average loss: 0.4424, Accuracy: 8495/10000 (84.95%)

EPOCH: 38
Loss=0.4916757345199585 Batch_id=781 Accuracy=78.09: 100%|██████████| 782/782 [00:21<00:00, 37.22it/s]

Test set: Average loss: 0.4284, Accuracy: 8550/10000 (85.50%)

EPOCH: 39
Loss=0.38577908277511597 Batch_id=781 Accuracy=78.51: 100%|██████████| 782/782 [00:20<00:00, 37.35it/s]

Test set: Average loss: 0.4357, Accuracy: 8500/10000 (85.00%)

EPOCH: 40
Loss=0.7176442742347717 Batch_id=781 Accuracy=78.44: 100%|██████████| 782/782 [00:20<00:00, 37.33it/s]

Test set: Average loss: 0.4296, Accuracy: 8533/10000 (85.33%)

EPOCH: 41
Loss=0.681721031665802 Batch_id=781 Accuracy=78.62: 100%|██████████| 782/782 [00:21<00:00, 37.16it/s]

Test set: Average loss: 0.4402, Accuracy: 8513/10000 (85.13%)

EPOCH: 42
Loss=0.30835509300231934 Batch_id=781 Accuracy=78.69: 100%|██████████| 782/782 [00:21<00:00, 36.86it/s]

Test set: Average loss: 0.4251, Accuracy: 8583/10000 (85.83%)

EPOCH: 43
Loss=0.8883230090141296 Batch_id=781 Accuracy=78.76: 100%|██████████| 782/782 [00:21<00:00, 36.95it/s]

Test set: Average loss: 0.4295, Accuracy: 8551/10000 (85.51%)

EPOCH: 44
Loss=0.9213531017303467 Batch_id=781 Accuracy=79.01: 100%|██████████| 782/782 [00:21<00:00, 36.63it/s]

Test set: Average loss: 0.4193, Accuracy: 8581/10000 (85.81%)

EPOCH: 45
Loss=0.6835362911224365 Batch_id=781 Accuracy=78.87: 100%|██████████| 782/782 [00:21<00:00, 35.89it/s]

Test set: Average loss: 0.4164, Accuracy: 8606/10000 (86.06%)

EPOCH: 46
Loss=1.0568721294403076 Batch_id=781 Accuracy=79.19: 100%|██████████| 782/782 [00:22<00:00, 35.25it/s]

Test set: Average loss: 0.4101, Accuracy: 8645/10000 (86.45%)

EPOCH: 47
Loss=0.8965411186218262 Batch_id=781 Accuracy=79.29: 100%|██████████| 782/782 [00:22<00:00, 35.32it/s]

Test set: Average loss: 0.4255, Accuracy: 8563/10000 (85.63%)

EPOCH: 48
Loss=0.8628125786781311 Batch_id=781 Accuracy=79.57: 100%|██████████| 782/782 [00:21<00:00, 35.96it/s]

Test set: Average loss: 0.4228, Accuracy: 8564/10000 (85.64%)

EPOCH: 49
Loss=0.7743484973907471 Batch_id=781 Accuracy=79.45: 100%|██████████| 782/782 [00:21<00:00, 36.24it/s]

Test set: Average loss: 0.4148, Accuracy: 8602/10000 (86.02%)


```

# Training & Test Accuracy

<img width="890" alt="image" src="https://user-images.githubusercontent.com/57046534/218267911-92f49dd1-6856-4e41-b451-61cd804b7ad1.png">

# Mis-Classified Images

![image](https://user-images.githubusercontent.com/57046534/218391439-7deca60a-382d-4dce-886c-ad76a55ea71f.png)



