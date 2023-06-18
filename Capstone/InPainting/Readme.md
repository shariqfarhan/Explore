# Inpainting Training

## Summary

We use CIFAR10 as the base dataset for inpainting. Leveraging the cutout (Coarse Dropout) augmentation from albumentations library, we create a dataset comprising of base image (from CIFAR10) and the masked image (with Cutout augmentation). Then a UNET model is then trained to minimize the loss between the base image and the masked image.


