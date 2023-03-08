# Assignment Submission for Session 7 - Advanced Training Concepts

Assignment: 

1. Check this Repo out: https://github.com/kuangliu/pytorch-cifar
2. You are going to follow the same structure for your Code from now on. So Create:
    1. models folder - this is where you'll add all of your future models. Copy resnet.py into this folder, this file should only have ResNet 18/34 models. Delete Bottleneck Class
    2. main.py - from Google Colab, now onwards, this is the file that you'll import (along with the model). Your main file shall be able to take these params or you should be able to pull functions from it and then perform operations, like (including but not limited to):
          1. training and test loops
          2. data split between test and train
          3. epochs
          4. batch size
          5. which optimizer to run
          6. do we run a scheduler?
    3. utils.py file (or a folder later on when it expands) - this is where you will add all of your utilities like:
          1. image transforms,
          2. gradcam,
          3. misclassification code,
          4. tensorboard related stuff
          5. advanced training policies, etc etc
    4. Name this main repos something, and don't call it Assignment 7. This is what you'll import for all the rest of the assignments. Add a proper readme describing all the files. 
3. Your assignment is to build the above training structure. Train ResNet18 on Cifar10 for 20 Epochs. The assignment must:
      1. pull your Github code to google colab (don't copy-paste code)
      2. prove that you are following the above structure
      3. that the code in your google collab notebook is NOTHING.. barely anything. There should not be any function or class that you can define in your Google Colab Notebook. Everything must be imported from all of your other files
      4. your colab file must:
          1. train resnet18 for 20 epochs on the CIFAR10 dataset
          2. show loss curves for test and train datasets
          3. show a gallery of 10 misclassified images
          4. show gradcamLinks to an external site. output on 10 misclassified images. Remember if you are applying GradCAM on a channel that is less than 5px, then please don't bother to submit the assignment. 😡🤬🤬🤬🤬
          5. Once done, upload the code to GitHub, and share the code. This readme must link to the main repo so we can read your file structure. 
          6. Train for 20 epochs
          7. Get 10 misclassified images
          8. Get 10 GradCam outputs on any misclassified images (remember that you MUST use the library we discussed in the class)
          9. Apply these transforms while training:
              1. RandomCrop(32, padding=4)
              2. CutOut(16x16)

4. Assignment Submission Questions:
        1. Share the COMPLETE code of your model.py
        2. Share the COMPLETE code of your utils.py
        3. Share the COMPLETE code of your main.py
        4. Copy-paste the training log (cannot be ugly)
        5. Copy-paste the 10/20 Misclassified Images Gallery
        6. Copy-paste the 10/20 GradCam outputs Gallery
        7. Share the link to your MAIN repo
        8. Share the link to your README of Assignment 7 (cannot be in the MAIN Repo, but Assignment 8 repo)



# Solution
Please find the reference notebook [Assignment 7 - GradCAM on CIFAR](https://github.com/shariqfarhan/Explore/blob/master/Assignment_7/Assignment_7_Final.ipynb) 

## Summary

1. Total Model Parameters :  11,173,962
2. Best Train Accuracy :  85.9375
3. Best Test Accuracy :  85.42
4. Number of Epochs with test accuracy above 85% threshold: 2

## Accuracy & Loss Curves for Test & Train Dataset

![image](https://user-images.githubusercontent.com/57046534/223698589-801193d2-65ca-493d-8979-d6e7b5015914.png)



## Misclassified Images

![image](https://user-images.githubusercontent.com/57046534/223698616-4b645bc3-4d34-46be-9492-8c33ef4b58e7.png)


## GradCAM on 10 misclassified Images
GradCAM in Layer 4

![image](https://user-images.githubusercontent.com/57046534/223698642-f39366c2-b693-487d-b2e6-c7043248c38c.png)


## Training Logs

```
EPOCH: 1 (LR: 0.01)
Loss=2.4372575283050537 Batch_id=781 Accuracy=38.18: 100%|██████████| 782/782 [00:37<00:00, 21.03it/s]

Test set: Average loss: -3.3378, Accuracy: 5802/10000 (58.02%)

EPOCH: 2 (LR: 0.01)
Loss=2.0401816368103027 Batch_id=781 Accuracy=52.83: 100%|██████████| 782/782 [00:37<00:00, 20.74it/s]

Test set: Average loss: -5.0625, Accuracy: 6825/10000 (68.25%)

EPOCH: 3 (LR: 0.01)
Loss=1.8356696367263794 Batch_id=781 Accuracy=59.70: 100%|██████████| 782/782 [00:37<00:00, 20.83it/s]

Test set: Average loss: -5.1034, Accuracy: 7265/10000 (72.65%)

EPOCH: 4 (LR: 0.01)
Loss=2.1858654022216797 Batch_id=781 Accuracy=63.28: 100%|██████████| 782/782 [00:37<00:00, 20.83it/s]

Test set: Average loss: -6.1370, Accuracy: 7395/10000 (73.95%)

EPOCH: 5 (LR: 0.01)
Loss=1.3729472160339355 Batch_id=781 Accuracy=66.28: 100%|██████████| 782/782 [00:37<00:00, 20.71it/s]

Test set: Average loss: -6.6055, Accuracy: 7495/10000 (74.95%)

EPOCH: 6 (LR: 0.01)
Loss=1.837242841720581 Batch_id=781 Accuracy=68.66: 100%|██████████| 782/782 [00:37<00:00, 20.98it/s] 

Test set: Average loss: -7.0673, Accuracy: 7930/10000 (79.30%)

EPOCH: 7 (LR: 0.01)
Loss=2.218010663986206 Batch_id=781 Accuracy=69.75: 100%|██████████| 782/782 [00:37<00:00, 20.70it/s] 

Test set: Average loss: -7.2671, Accuracy: 7872/10000 (78.72%)

EPOCH: 8 (LR: 0.01)
Loss=1.3488136529922485 Batch_id=781 Accuracy=71.54: 100%|██████████| 782/782 [00:37<00:00, 20.97it/s]

Test set: Average loss: -7.6757, Accuracy: 8201/10000 (82.01%)

EPOCH: 9 (LR: 0.01)
Loss=1.7038609981536865 Batch_id=781 Accuracy=72.36: 100%|██████████| 782/782 [00:37<00:00, 20.78it/s]

Test set: Average loss: -7.7558, Accuracy: 8211/10000 (82.11%)

EPOCH: 10 (LR: 0.01)
Loss=1.4642369747161865 Batch_id=781 Accuracy=73.21: 100%|██████████| 782/782 [00:37<00:00, 20.65it/s]

Test set: Average loss: -7.7845, Accuracy: 8229/10000 (82.29%)

EPOCH: 11 (LR: 0.01)
Loss=1.8160532712936401 Batch_id=781 Accuracy=74.17: 100%|██████████| 782/782 [00:38<00:00, 20.56it/s]

Test set: Average loss: -8.0763, Accuracy: 8297/10000 (82.97%)

EPOCH: 12 (LR: 0.01)
Loss=1.4931230545043945 Batch_id=781 Accuracy=74.92: 100%|██████████| 782/782 [00:37<00:00, 20.91it/s]

Test set: Average loss: -8.5894, Accuracy: 8300/10000 (83.00%)

EPOCH: 13 (LR: 0.01)
Loss=1.588808298110962 Batch_id=781 Accuracy=75.60: 100%|██████████| 782/782 [00:37<00:00, 20.67it/s] 

Test set: Average loss: -8.6215, Accuracy: 8101/10000 (81.01%)

EPOCH: 14 (LR: 0.01)
Loss=1.744088888168335 Batch_id=781 Accuracy=76.57: 100%|██████████| 782/782 [00:37<00:00, 20.94it/s] 

Test set: Average loss: -9.6370, Accuracy: 8345/10000 (83.45%)

EPOCH: 15 (LR: 0.01)
Loss=1.4941519498825073 Batch_id=781 Accuracy=76.67: 100%|██████████| 782/782 [00:37<00:00, 20.66it/s]

Test set: Average loss: -9.1359, Accuracy: 8484/10000 (84.84%)

EPOCH: 16 (LR: 0.01)
Loss=1.3103077411651611 Batch_id=781 Accuracy=77.18: 100%|██████████| 782/782 [00:37<00:00, 20.93it/s]

Test set: Average loss: -9.5423, Accuracy: 8219/10000 (82.19%)

EPOCH: 17 (LR: 0.01)
Loss=0.9820313453674316 Batch_id=781 Accuracy=77.56: 100%|██████████| 782/782 [00:37<00:00, 20.82it/s]

Test set: Average loss: -9.3675, Accuracy: 8542/10000 (85.42%)

EPOCH: 18 (LR: 0.01)
Loss=1.1647119522094727 Batch_id=781 Accuracy=78.11: 100%|██████████| 782/782 [00:37<00:00, 20.64it/s]

Test set: Average loss: -9.1116, Accuracy: 8482/10000 (84.82%)

EPOCH: 19 (LR: 0.01)
Loss=1.4211103916168213 Batch_id=781 Accuracy=78.09: 100%|██████████| 782/782 [00:37<00:00, 20.91it/s]

Test set: Average loss: -9.3036, Accuracy: 8366/10000 (83.66%)

EPOCH: 20 (LR: 0.01)
Loss=0.7216936349868774 Batch_id=781 Accuracy=78.69: 100%|██████████| 782/782 [00:38<00:00, 20.53it/s]

Test set: Average loss: -9.1363, Accuracy: 8527/10000 (85.27%)

```


