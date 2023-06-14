# Part 1 : UNet from Scratch

## Part 1 Summary
This repository contains the code for building a UNet model from scratch using PyTorch. The UNet architecture is a popular and effective model for image segmentation tasks. In this project, we have trained the UNet model four times, each time using a different approach for downsampling and upsampling, as well as different loss functions.

Model Architectures
The following model architectures were implemented and trained:

UNet with Max Pooling: This model uses max pooling for downsampling and upsampling operations. It is a commonly used approach in UNet implementations.

UNet with Strided Convolution: Instead of using max pooling, this model utilizes strided convolutions for downsampling and transposed convolutions for upsampling.

UNet with Upsampling: This model uses simple upsampling and convolutional layers for upsampling instead of transposed convolutions.

Loss Functions
The following loss functions were used during training:

Binary Cross Entropy (BCE) Loss: This loss function is commonly used for binary segmentation tasks, where each pixel is classified as foreground or background.

Dice Loss: The Dice coefficient is a similarity metric commonly used in image segmentation tasks. This loss function combines the BCE loss and the Dice coefficient to optimize both the pixel-level accuracy and the overall similarity of the segmentation.


Results
After training the UNet model using different architectures and loss functions, we achieved the following results:

Model 1 (Max Pooling, Transpose, BCE Loss): IOU of 0.58.

Model 2 (Max Pooling, Transpose, Dice Loss): IOU of 0.58.

Model 3 (Strided Convolution, Transpose, BCE Loss): IOU of 0.58.

Model 4 (Strided Convolution, UpSampling, Dice Loss): IOU of 0.58.

Please refer to the training logs in the detailed section in the end

# Part 2 : MNIST VAE

The code for this can be found in [this file](https://github.com/shariqfarhan/Explore/blob/master/Assignment_13/VAE%20-%20MNIST.ipynb)

A sample of 64 images where MNIST image is passed to the VAE model with correct label

[Correct Label](https://github.com/shariqfarhan/Explore/blob/master/Assignment_13/MNIST_Correct_label.png)

![MNIST_Correct_label](https://github.com/shariqfarhan/Explore/assets/57046534/b84b2502-77f3-46da-a39f-3723ce62cb88)



A sample of 64 images where MNIST image is passed to the VAE model with incorrect label

[Incorrect Label](https://github.com/shariqfarhan/Explore/blob/master/Assignment_13/MNIST_incorrect_label.png)
![MNIST_incorrect_label](https://github.com/shariqfarhan/Explore/assets/57046534/f5710f08-ecb0-4364-9153-3e52e7f2a0ce)


# Part 3 : CIFAR VAE

The code for this can be found in [this file](https://github.com/shariqfarhan/Explore/blob/master/Assignment_13/VAE%20-%20CIFAR.ipynb)

# Part 1 : UNet from Scratch - Detailed

# Model 1 Output Logs

```
Epoch: 1. Train.      Loss: 0.616: 100%|██████████| 375/375 [01:07<00:00,  5.54it/s]
Epoch: 1. Validation. Loss: 0.560: 100%|██████████| 86/86 [00:06<00:00, 13.13it/s]
Epoch: 2. Train.      Loss: 0.584: 100%|██████████| 375/375 [01:07<00:00,  5.52it/s]
Epoch: 2. Validation. Loss: 0.552: 100%|██████████| 86/86 [00:05<00:00, 15.98it/s]
Epoch: 3. Train.      Loss: 0.564: 100%|██████████| 375/375 [01:08<00:00,  5.51it/s]
Epoch: 3. Validation. Loss: 0.537: 100%|██████████| 86/86 [00:05<00:00, 15.78it/s]
Epoch: 4. Train.      Loss: 0.556: 100%|██████████| 375/375 [01:08<00:00,  5.51it/s]
Epoch: 4. Validation. Loss: 0.535: 100%|██████████| 86/86 [00:05<00:00, 16.48it/s]
Epoch: 5. Train.      Loss: 0.546: 100%|██████████| 375/375 [01:08<00:00,  5.51it/s]
Epoch: 5. Validation. Loss: 0.522: 100%|██████████| 86/86 [00:05<00:00, 16.51it/s]
Epoch: 6. Train.      Loss: 0.540: 100%|██████████| 375/375 [01:08<00:00,  5.51it/s]
Epoch: 6. Validation. Loss: 0.518: 100%|██████████| 86/86 [00:05<00:00, 16.12it/s]
Epoch: 7. Train.      Loss: 0.534: 100%|██████████| 375/375 [01:08<00:00,  5.51it/s]
Epoch: 7. Validation. Loss: 0.511: 100%|██████████| 86/86 [00:05<00:00, 16.08it/s]
Epoch: 8. Train.      Loss: 0.526: 100%|██████████| 375/375 [01:08<00:00,  5.51it/s]
Epoch: 8. Validation. Loss: 0.512: 100%|██████████| 86/86 [00:05<00:00, 16.38it/s]
Epoch: 9. Train.      Loss: 0.521: 100%|██████████| 375/375 [01:08<00:00,  5.51it/s]
Epoch: 9. Validation. Loss: 0.505: 100%|██████████| 86/86 [00:05<00:00, 16.50it/s]
Epoch: 10. Train.      Loss: 0.521: 100%|██████████| 375/375 [01:07<00:00,  5.53it/s]
Epoch: 10. Validation. Loss: 0.500: 100%|██████████| 86/86 [00:05<00:00, 16.39it/s]
Epoch: 11. Train.      Loss: 0.515: 100%|██████████| 375/375 [01:07<00:00,  5.52it/s]
Epoch: 11. Validation. Loss: 0.494: 100%|██████████| 86/86 [00:05<00:00, 15.71it/s]
Epoch: 12. Train.      Loss: 0.516: 100%|██████████| 375/375 [01:08<00:00,  5.50it/s]
Epoch: 12. Validation. Loss: 0.500: 100%|██████████| 86/86 [00:05<00:00, 16.07it/s]
Epoch: 13. Train.      Loss: 0.512: 100%|██████████| 375/375 [01:07<00:00,  5.52it/s]
Epoch: 13. Validation. Loss: 0.496: 100%|██████████| 86/86 [00:05<00:00, 16.54it/s]
Epoch: 14. Train.      Loss: 0.511: 100%|██████████| 375/375 [01:07<00:00,  5.52it/s]
Epoch: 14. Validation. Loss: 0.492: 100%|██████████| 86/86 [00:05<00:00, 15.76it/s]
Epoch: 15. Train.      Loss: 0.505: 100%|██████████| 375/375 [01:07<00:00,  5.52it/s]
Epoch: 15. Validation. Loss: 0.492: 100%|██████████| 86/86 [00:05<00:00, 16.17it/s]
Epoch: 16. Train.      Loss: 0.506: 100%|██████████| 375/375 [01:07<00:00,  5.52it/s]
Epoch: 16. Validation. Loss: 0.495: 100%|██████████| 86/86 [00:05<00:00, 16.42it/s]
Epoch: 17. Train.      Loss: 0.500: 100%|██████████| 375/375 [01:07<00:00,  5.52it/s]
Epoch: 17. Validation. Loss: 0.488: 100%|██████████| 86/86 [00:05<00:00, 15.72it/s]
Epoch: 18. Train.      Loss: 0.499: 100%|██████████| 375/375 [01:08<00:00,  5.50it/s]
Epoch: 18. Validation. Loss: 0.490: 100%|██████████| 86/86 [00:05<00:00, 15.32it/s]
Epoch: 19. Train.      Loss: 0.497: 100%|██████████| 375/375 [01:07<00:00,  5.52it/s]
Epoch: 19. Validation. Loss: 0.488: 100%|██████████| 86/86 [00:05<00:00, 16.44it/s]
Epoch: 20. Train.      Loss: 0.493: 100%|██████████| 375/375 [01:08<00:00,  5.50it/s]
Epoch: 20. Validation. Loss: 0.488: 100%|██████████| 86/86 [00:05<00:00, 16.35it/s]

```
# Model 2 Output Logs

```
Epoch: 1. Train.      Loss: 0.670: 100%|██████████| 375/375 [01:07<00:00,  5.58it/s]
Epoch: 1. Validation. Loss: 0.652: 100%|██████████| 86/86 [00:05<00:00, 16.29it/s]
Epoch: 2. Train.      Loss: 0.639: 100%|██████████| 375/375 [01:07<00:00,  5.55it/s]
Epoch: 2. Validation. Loss: 0.633: 100%|██████████| 86/86 [00:05<00:00, 15.93it/s]
Epoch: 3. Train.      Loss: 0.620: 100%|██████████| 375/375 [01:07<00:00,  5.53it/s]
Epoch: 3. Validation. Loss: 0.599: 100%|██████████| 86/86 [00:05<00:00, 16.43it/s]
Epoch: 4. Train.      Loss: 0.603: 100%|██████████| 375/375 [01:07<00:00,  5.53it/s]
Epoch: 4. Validation. Loss: 0.594: 100%|██████████| 86/86 [00:05<00:00, 16.54it/s]
Epoch: 5. Train.      Loss: 0.591: 100%|██████████| 375/375 [01:07<00:00,  5.52it/s]
Epoch: 5. Validation. Loss: 0.576: 100%|██████████| 86/86 [00:05<00:00, 16.43it/s]
Epoch: 6. Train.      Loss: 0.578: 100%|██████████| 375/375 [01:08<00:00,  5.50it/s]
Epoch: 6. Validation. Loss: 0.555: 100%|██████████| 86/86 [00:05<00:00, 15.99it/s]
Epoch: 7. Train.      Loss: 0.568: 100%|██████████| 375/375 [01:07<00:00,  5.53it/s]
Epoch: 7. Validation. Loss: 0.549: 100%|██████████| 86/86 [00:05<00:00, 15.83it/s]
Epoch: 8. Train.      Loss: 0.559: 100%|██████████| 375/375 [01:08<00:00,  5.50it/s]
Epoch: 8. Validation. Loss: 0.541: 100%|██████████| 86/86 [00:05<00:00, 16.42it/s]
Epoch: 9. Train.      Loss: 0.551: 100%|██████████| 375/375 [01:07<00:00,  5.53it/s]
Epoch: 9. Validation. Loss: 0.536: 100%|██████████| 86/86 [00:05<00:00, 15.88it/s]
Epoch: 10. Train.      Loss: 0.544: 100%|██████████| 375/375 [01:07<00:00,  5.53it/s]
Epoch: 10. Validation. Loss: 0.520: 100%|██████████| 86/86 [00:05<00:00, 16.47it/s]
Epoch: 11. Train.      Loss: 0.537: 100%|██████████| 375/375 [01:07<00:00,  5.53it/s]
Epoch: 11. Validation. Loss: 0.515: 100%|██████████| 86/86 [00:05<00:00, 16.43it/s]
Epoch: 12. Train.      Loss: 0.532: 100%|██████████| 375/375 [01:07<00:00,  5.53it/s]
Epoch: 12. Validation. Loss: 0.514: 100%|██████████| 86/86 [00:05<00:00, 16.00it/s]
Epoch: 13. Train.      Loss: 0.527: 100%|██████████| 375/375 [01:08<00:00,  5.51it/s]
Epoch: 13. Validation. Loss: 0.509: 100%|██████████| 86/86 [00:05<00:00, 15.66it/s]
Epoch: 14. Train.      Loss: 0.523: 100%|██████████| 375/375 [01:07<00:00,  5.52it/s]
Epoch: 14. Validation. Loss: 0.512: 100%|██████████| 86/86 [00:05<00:00, 16.44it/s]
Epoch: 15. Train.      Loss: 0.518: 100%|██████████| 375/375 [01:08<00:00,  5.51it/s]
Epoch: 15. Validation. Loss: 0.503: 100%|██████████| 86/86 [00:05<00:00, 15.84it/s]
Epoch: 16. Train.      Loss: 0.515: 100%|██████████| 375/375 [01:08<00:00,  5.51it/s]
Epoch: 16. Validation. Loss: 0.509: 100%|██████████| 86/86 [00:05<00:00, 16.63it/s]
Epoch: 17. Train.      Loss: 0.514: 100%|██████████| 375/375 [01:07<00:00,  5.53it/s]
Epoch: 17. Validation. Loss: 0.579: 100%|██████████| 86/86 [00:05<00:00, 15.97it/s]
Epoch: 18. Train.      Loss: 0.510: 100%|██████████| 375/375 [01:07<00:00,  5.52it/s]
Epoch: 18. Validation. Loss: 0.517: 100%|██████████| 86/86 [00:05<00:00, 16.49it/s]
Epoch: 19. Train.      Loss: 0.515: 100%|██████████| 375/375 [01:07<00:00,  5.52it/s]
Epoch: 19. Validation. Loss: 0.497: 100%|██████████| 86/86 [00:05<00:00, 16.55it/s]
Epoch: 20. Train.      Loss: 0.504: 100%|██████████| 375/375 [01:07<00:00,  5.52it/s]
Epoch: 20. Validation. Loss: 0.496: 100%|██████████| 86/86 [00:05<00:00, 16.61it/s]
```

# Base Data Ground Truth

![image](https://user-images.githubusercontent.com/57046534/232242181-72a2f0ff-04a4-49e7-b91a-e50a9574c143.png)


# Predictions with UNet

This was run for 20 Epochs with Max Pooling, DeConvolution & BCE Loss

![image](https://user-images.githubusercontent.com/57046534/232242161-c6f3c137-fd74-40d3-a074-bb7318038548.png)

Model 2 Ouptut

![image](https://github.com/shariqfarhan/Explore/assets/57046534/c91c8fec-a2e4-40c4-9b4b-531c4b5c1ee1)

```
100%|██████████| 86/86 [00:10<00:00,  7.99it/s]
Model 1 IOU score: 0.58
100%|██████████| 86/86 [00:10<00:00,  8.07it/s]
Model 2 IOU score: 0.58
100%|██████████| 86/86 [00:05<00:00, 15.41it/s]
Model 3 IOU score: 0.58
100%|██████████| 86/86 [00:05<00:00, 14.96it/s]
Model 4 IOU score: 0.58
```

References

1. [Albumentations library implementation of Semantic Segmentation](https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/)
2. [U-Net](https://nn.labml.ai/unet/index.html)
