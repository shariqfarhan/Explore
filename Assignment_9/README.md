# The Assignment

Build the following network:

1.  That takes a CIFAR10 image (32x32x3)
2.  Add 3 Convolutions to arrive at AxAx48 dimensions (e.g. 32x32x3 | 3x3x3x16 >> 3x3x16x32 >> 3x3x32x48)
3.  Apply GAP and get 1x1x48, call this X
4.  Create a block called ULTIMUS that:
    1. Creates 3 FC layers called K, Q and V such that:
        1. X*K = 48*48x8 > 8
        2. X*Q = 48*48x8 > 8 
        3. X*V = 48*48x8 > 8 
    2. then create AM = SoftMax(QTK)/(8^0.5) = 8*8 = 8
    3. then Z = V*AM = 8*8 > 8
    4. then another FC layer called Out that:
       1. Z*Out = 8*8x48 > 48
5. Repeat this Ultimus block 4 times
6. Then add final FC layer that converts 48 to 10 and sends it to the loss function.
7. Model would look like this C>C>C>U>U>U>U>FFC>Loss
8. Train the model for 24 epochs using the OCP that I wrote in class. Use ADAM as an optimizer. 
9. Submit the link and answer the questions on the assignment page:
    1. Share the link to the main repo (must have Assignment 7/8/9 model7/8/9.py files (or similarly named))
    2. Share the code of model9.py
    3. Copy and paste the Training Log
    4. Copy and paste the training and validation loss chart


# Solution

The Assignment involved building a neural network with a custom block called `Ultimus`, in this we calculate the attention matrix of given inputs of Query, Key & Value.

We train the neural network for 24 epochs using One Cycle Policy. The below graphs show the model performance during the traing & test phase.
The detailed solution & working can be found [here](https://github.com/shariqfarhan/Explore/blob/master/Assignment_9/Assignment_9.ipynb)

## Model Performance Summary

```
Total Model Parameters :  21154
Best Train Accuracy :  59.375
Best Test Accuracy :  51.58
```

## Model Train & Test loss & Accuracy
![image](https://user-images.githubusercontent.com/57046534/227075859-63729d1e-9333-46ce-90e6-eeac6cebdb47.png)


## Model Training logs
```
EPOCH: 1 (LR: 0.0014449370798837726)
Loss=2.323298454284668 Batch_id=781 Accuracy=9.71: 100%|██████████| 782/782 [00:09<00:00, 82.42it/s]  

Test set: Average loss: 2.3035, Accuracy: 1000/10000 (10.00%)

EPOCH: 2 (LR: 0.008382409352909775)
Loss=2.310532808303833 Batch_id=781 Accuracy=9.85: 100%|██████████| 782/782 [00:09<00:00, 79.01it/s] 

Test set: Average loss: 2.3030, Accuracy: 1000/10000 (10.00%)

EPOCH: 3 (LR: 0.01531988162593578)
Loss=2.310706853866577 Batch_id=781 Accuracy=9.96: 100%|██████████| 782/782 [00:09<00:00, 82.06it/s]  

Test set: Average loss: 2.3034, Accuracy: 1000/10000 (10.00%)

EPOCH: 4 (LR: 0.022257353898961784)
Loss=2.30141544342041 Batch_id=781 Accuracy=9.63: 100%|██████████| 782/782 [00:09<00:00, 82.86it/s]  

Test set: Average loss: 2.3030, Accuracy: 1000/10000 (10.00%)

EPOCH: 5 (LR: 0.029194826171987785)
Loss=2.284342050552368 Batch_id=781 Accuracy=9.84: 100%|██████████| 782/782 [00:09<00:00, 79.38it/s]  

Test set: Average loss: 2.3038, Accuracy: 1000/10000 (10.00%)

EPOCH: 6 (LR: 0.036120995762573965)
Loss=2.3111190795898438 Batch_id=781 Accuracy=10.09: 100%|██████████| 782/782 [00:09<00:00, 82.68it/s]

Test set: Average loss: 2.3032, Accuracy: 1000/10000 (10.00%)

EPOCH: 7 (LR: 0.03421977036765889)
Loss=2.304633378982544 Batch_id=781 Accuracy=10.02: 100%|██████████| 782/782 [00:09<00:00, 82.33it/s] 

Test set: Average loss: 2.3029, Accuracy: 1000/10000 (10.00%)

EPOCH: 8 (LR: 0.03231854497274383)
Loss=2.3115153312683105 Batch_id=781 Accuracy=9.71: 100%|██████████| 782/782 [00:09<00:00, 80.43it/s]

Test set: Average loss: 2.3043, Accuracy: 1000/10000 (10.00%)

EPOCH: 9 (LR: 0.030417319577828755)
Loss=2.3101935386657715 Batch_id=781 Accuracy=10.52: 100%|██████████| 782/782 [00:09<00:00, 79.85it/s]

Test set: Average loss: 2.2492, Accuracy: 1746/10000 (17.46%)

EPOCH: 10 (LR: 0.028516094182913686)
Loss=2.0955066680908203 Batch_id=781 Accuracy=19.24: 100%|██████████| 782/782 [00:09<00:00, 82.03it/s]

Test set: Average loss: 2.0631, Accuracy: 1835/10000 (18.35%)

EPOCH: 11 (LR: 0.026614868787998618)
Loss=1.6953871250152588 Batch_id=781 Accuracy=22.50: 100%|██████████| 782/782 [00:09<00:00, 79.92it/s]

Test set: Average loss: 1.8389, Accuracy: 2588/10000 (25.88%)

EPOCH: 12 (LR: 0.024713643393083545)
Loss=2.3508996963500977 Batch_id=781 Accuracy=26.54: 100%|██████████| 782/782 [00:09<00:00, 79.18it/s]

Test set: Average loss: 1.8409, Accuracy: 2442/10000 (24.42%)

EPOCH: 13 (LR: 0.022812417998168476)
Loss=1.5601086616516113 Batch_id=781 Accuracy=28.82: 100%|██████████| 782/782 [00:09<00:00, 79.39it/s]

Test set: Average loss: 1.6397, Accuracy: 3387/10000 (33.87%)

EPOCH: 14 (LR: 0.020911192603253408)
Loss=1.2927627563476562 Batch_id=781 Accuracy=32.47: 100%|██████████| 782/782 [00:09<00:00, 78.62it/s]

Test set: Average loss: 1.6988, Accuracy: 3408/10000 (34.08%)

EPOCH: 15 (LR: 0.01900996720833834)
Loss=1.6733932495117188 Batch_id=781 Accuracy=33.66: 100%|██████████| 782/782 [00:09<00:00, 81.39it/s]

Test set: Average loss: 1.5977, Accuracy: 3618/10000 (36.18%)

EPOCH: 16 (LR: 0.017108741813423266)
Loss=1.660379409790039 Batch_id=781 Accuracy=28.63: 100%|██████████| 782/782 [00:09<00:00, 80.94it/s] 

Test set: Average loss: 1.7282, Accuracy: 3240/10000 (32.40%)

EPOCH: 17 (LR: 0.015207516418508198)
Loss=1.7486686706542969 Batch_id=781 Accuracy=33.93: 100%|██████████| 782/782 [00:09<00:00, 82.24it/s]

Test set: Average loss: 1.5617, Accuracy: 3617/10000 (36.17%)

EPOCH: 18 (LR: 0.013306291023593125)
Loss=1.6327943801879883 Batch_id=781 Accuracy=35.90: 100%|██████████| 782/782 [00:09<00:00, 82.30it/s]

Test set: Average loss: 1.6336, Accuracy: 3669/10000 (36.69%)

EPOCH: 19 (LR: 0.01140506562867806)
Loss=1.4548964500427246 Batch_id=781 Accuracy=38.21: 100%|██████████| 782/782 [00:09<00:00, 80.48it/s]

Test set: Average loss: 1.5016, Accuracy: 4127/10000 (41.27%)

EPOCH: 20 (LR: 0.009503840233762988)
Loss=1.5125818252563477 Batch_id=781 Accuracy=40.60: 100%|██████████| 782/782 [00:10<00:00, 77.37it/s]

Test set: Average loss: 1.4506, Accuracy: 4348/10000 (43.48%)

EPOCH: 21 (LR: 0.007602614838847919)
Loss=1.9447730779647827 Batch_id=781 Accuracy=41.50: 100%|██████████| 782/782 [00:09<00:00, 81.14it/s]

Test set: Average loss: 1.4883, Accuracy: 4385/10000 (43.85%)

EPOCH: 22 (LR: 0.00570138944393285)
Loss=1.9937145709991455 Batch_id=781 Accuracy=43.57: 100%|██████████| 782/782 [00:09<00:00, 79.86it/s]

Test set: Average loss: 1.4136, Accuracy: 4621/10000 (46.21%)

EPOCH: 23 (LR: 0.0038001640490177777)
Loss=1.393161416053772 Batch_id=781 Accuracy=45.62: 100%|██████████| 782/782 [00:09<00:00, 81.99it/s] 

Test set: Average loss: 1.3570, Accuracy: 4784/10000 (47.84%)

EPOCH: 24 (LR: 0.0018989386541027123)
Loss=1.9116569757461548 Batch_id=781 Accuracy=47.76: 100%|██████████| 782/782 [00:09<00:00, 80.83it/s]

Test set: Average loss: 1.3424, Accuracy: 4914/10000 (49.14%)
```
