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


# Receptive Field Calculation for the model

```
Input_features	padding	kernel	stride	dilation	jump_In	jump_out	RF_In	RF_Out	Output_Feature	Input_Channel	Output_Channels	kernel_adjusted	Convolution Type
32	2	3	1	2	1	1	1	5	32	3	16	5	Dilated Conv
32	1	3	1	1	1	1	5	7	32	16	32	3	Normal
32	1	3	1	1	1	1	7	9	32	32	64	3	Normal
32	1	3	1	1	1	1	9	11	32	64	64	3	Normal
32	2	3	2	2	1	2	9	13	16	64	64	5	Dilated + Strided Conv
16	2	3	1	2	2	2	13	21	16	64	128	5	Depthwise
16	0	1	1	1	2	2	21	21	16	128	128	1	Pointwise Conv
16	2	3	1	2	2	2	21	29	16	128	256	5	Depthwise
16	0	1	1	1	2	2	29	29	16	256	256	1	Pointwise Conv
16	2	3	2	2	2	4	29	37	8	512	512	5	Dilated + Strided Conv
8	1	3	1	1	4	4	37	45	8	512	256	3	Depthwise
8	0	1	1	1	4	4	45	45	8	256	128	1	Pointwise Conv
8	1	3	1	1	4	4	45	53	8	128	64	3	Depthwise
8	0	1	1	2	4	4	53	53	8	64	64	1	Pointwise Conv
8	2	3	2	2	4	8	53	69	4	64	64	5	Dilated + Strided Conv
4	1	3	1	1	8	8	69	85	4	64	64	3	Depthwise
4	0	1	1	1	8	8	85	85	4	64	64	1	Pointwise Conv
4	2	3	1	2	8	8	85	117	4	64	64	5	Dilated Conv
4	1	3	1	1	8	8	117	133	4	64	64	3	Normal 


```

<img width="1061" alt="image" src="https://user-images.githubusercontent.com/57046534/218172835-4caae1d1-7165-441e-b206-affe9ccef940.png">

# Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
              ReLU-2           [-1, 16, 32, 32]               0
       BatchNorm2d-3           [-1, 16, 32, 32]              32
            Conv2d-4           [-1, 32, 32, 32]           4,608
              ReLU-5           [-1, 32, 32, 32]               0
       BatchNorm2d-6           [-1, 32, 32, 32]              64
            Conv2d-7           [-1, 64, 32, 32]          18,432
              ReLU-8           [-1, 64, 32, 32]               0
       BatchNorm2d-9           [-1, 64, 32, 32]             128
          Dropout-10           [-1, 64, 32, 32]               0
           Conv2d-11           [-1, 64, 32, 32]          36,864
             ReLU-12           [-1, 64, 32, 32]               0
      BatchNorm2d-13           [-1, 64, 32, 32]             128
          Dropout-14           [-1, 64, 32, 32]               0
           Conv2d-15           [-1, 64, 16, 16]          36,864
             ReLU-16           [-1, 64, 16, 16]               0
      BatchNorm2d-17           [-1, 64, 16, 16]             128
           Conv2d-18           [-1, 64, 16, 16]             576
           Conv2d-19           [-1, 64, 16, 16]           4,096
             ReLU-20           [-1, 64, 16, 16]               0
      BatchNorm2d-21           [-1, 64, 16, 16]             128
           Conv2d-22           [-1, 64, 16, 16]             576
           Conv2d-23           [-1, 32, 16, 16]           2,048
             ReLU-24           [-1, 32, 16, 16]               0
      BatchNorm2d-25           [-1, 32, 16, 16]              64
          Dropout-26           [-1, 32, 16, 16]               0
           Conv2d-27             [-1, 32, 8, 8]           9,216
             ReLU-28             [-1, 32, 8, 8]               0
      BatchNorm2d-29             [-1, 32, 8, 8]              64
          Dropout-30             [-1, 32, 8, 8]               0
           Conv2d-31             [-1, 32, 8, 8]             288
           Conv2d-32             [-1, 64, 8, 8]           2,048
             ReLU-33             [-1, 64, 8, 8]               0
      BatchNorm2d-34             [-1, 64, 8, 8]             128
          Dropout-35             [-1, 64, 8, 8]               0
           Conv2d-36             [-1, 64, 8, 8]             576
           Conv2d-37             [-1, 64, 8, 8]           4,096
             ReLU-38             [-1, 64, 8, 8]               0
      BatchNorm2d-39             [-1, 64, 8, 8]             128
          Dropout-40             [-1, 64, 8, 8]               0
           Conv2d-41             [-1, 64, 4, 4]          36,864
             ReLU-42             [-1, 64, 4, 4]               0
      BatchNorm2d-43             [-1, 64, 4, 4]             128
          Dropout-44             [-1, 64, 4, 4]               0
           Conv2d-45            [-1, 128, 4, 4]           1,152
           Conv2d-46             [-1, 64, 4, 4]           8,192
             ReLU-47             [-1, 64, 4, 4]               0
      BatchNorm2d-48             [-1, 64, 4, 4]             128
          Dropout-49             [-1, 64, 4, 4]               0
           Conv2d-50             [-1, 32, 4, 4]          18,432
             ReLU-51             [-1, 32, 4, 4]               0
      BatchNorm2d-52             [-1, 32, 4, 4]              64
          Dropout-53             [-1, 32, 4, 4]               0
        AvgPool2d-54             [-1, 32, 1, 1]               0
           Conv2d-55             [-1, 10, 1, 1]             320
================================================================
Total params: 186,992
Trainable params: 186,992
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.83
Params size (MB): 0.71
Estimated Total Size (MB): 7.55
----------------------------------------------------------------
```

# Training Logs

Unable to train in Colab for more than 12 epochs. If we can train for few more, we can hit the target of 85% in less than 200K params.

```
EPOCH: 0
Loss=1.752724051475525 Batch_id=1562 Accuracy=30.06: 100%|██████████| 1563/1563 [00:48<00:00, 31.97it/s]

Test set: Average loss: 1.6724, Accuracy: 3869/10000 (38.69%)

EPOCH: 1
Loss=1.8300813436508179 Batch_id=1562 Accuracy=38.57: 100%|██████████| 1563/1563 [00:41<00:00, 37.88it/s]

Test set: Average loss: 1.5662, Accuracy: 4243/10000 (42.43%)

EPOCH: 2
Loss=1.5969756841659546 Batch_id=1562 Accuracy=43.04: 100%|██████████| 1563/1563 [00:39<00:00, 39.12it/s]

Test set: Average loss: 1.4654, Accuracy: 4687/10000 (46.87%)

EPOCH: 3
Loss=1.6294562816619873 Batch_id=1562 Accuracy=46.06: 100%|██████████| 1563/1563 [00:40<00:00, 38.95it/s]

Test set: Average loss: 1.3918, Accuracy: 4909/10000 (49.09%)

EPOCH: 4
Loss=1.4020313024520874 Batch_id=1562 Accuracy=48.15: 100%|██████████| 1563/1563 [00:39<00:00, 39.52it/s]

Test set: Average loss: 1.3276, Accuracy: 5178/10000 (51.78%)

EPOCH: 5
Loss=2.0497381687164307 Batch_id=1562 Accuracy=50.34: 100%|██████████| 1563/1563 [00:39<00:00, 39.14it/s]

Test set: Average loss: 1.2618, Accuracy: 5455/10000 (54.55%)

EPOCH: 6
Loss=1.1431251764297485 Batch_id=1562 Accuracy=51.66: 100%|██████████| 1563/1563 [00:40<00:00, 38.66it/s]

Test set: Average loss: 1.2295, Accuracy: 5554/10000 (55.54%)

EPOCH: 7
Loss=1.2556120157241821 Batch_id=1562 Accuracy=53.61: 100%|██████████| 1563/1563 [00:39<00:00, 39.09it/s]

Test set: Average loss: 1.1780, Accuracy: 5774/10000 (57.74%)

EPOCH: 8
Loss=1.1045029163360596 Batch_id=1562 Accuracy=54.87: 100%|██████████| 1563/1563 [00:40<00:00, 38.75it/s]

Test set: Average loss: 1.1838, Accuracy: 5809/10000 (58.09%)

EPOCH: 9
Loss=1.7270148992538452 Batch_id=1562 Accuracy=55.93: 100%|██████████| 1563/1563 [00:40<00:00, 38.57it/s]

Test set: Average loss: 1.1555, Accuracy: 5929/10000 (59.29%)

EPOCH: 10
Loss=1.1595417261123657 Batch_id=1562 Accuracy=57.06: 100%|██████████| 1563/1563 [00:41<00:00, 38.08it/s]

Test set: Average loss: 1.1072, Accuracy: 6005/10000 (60.05%)

EPOCH: 11
Loss=1.2886956930160522 Batch_id=1562 Accuracy=58.15: 100%|██████████| 1563/1563 [00:39<00:00, 39.68it/s]

Test set: Average loss: 1.0765, Accuracy: 6188/10000 (61.88%)

EPOCH: 12
Loss=1.261199712753296 Batch_id=1562 Accuracy=59.47: 100%|██████████| 1563/1563 [00:39<00:00, 39.53it/s]

Test set: Average loss: 1.0669, Accuracy: 6192/10000 (61.92%)

EPOCH: 13
Loss=1.7568964958190918 Batch_id=1292 Accuracy=60.22:  83%|████████▎ | 1293/1563 [05:20<2:38:34, 35.24s/it]

```
