# import cv2
import torchvision
import torch
import torchvision.transforms as transforms

# Albumentations for augmentations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)


class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
      super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label


train_transforms = A.Compose([
            A.Sequential([
                A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
                A.PadIfNeeded(min_height=40, min_width=40),
                A.RandomCrop(32, 32),
                A.HorizontalFlip(p=1),
                A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8,
                                min_width=8, fill_value=(0.49139968, 0.48215841, 0.44653091), mask_fill_value=None, always_apply=True)
        ])
        ])

test_transforms = A.Compose([
    A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
      ToTensorV2(),
], p=1.0,
)


class args:
    def __init__(self, device="cpu", use_cuda=False) -> None:
        self.batch_size = 64
        self.device = device
        self.use_cuda = use_cuda
        self.kwargs = {"num_workers": 1, "pin_memory": True} if self.use_cuda else {}

trainset = Cifar10SearchDataset(
    root="./data", train=True, download=True, transform=train_transforms
)


trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args().batch_size, shuffle=True, **args().kwargs
)

testset = Cifar10SearchDataset(
    root="./data", train=False, download=True, transform=test_transforms
)


testloader = torch.utils.data.DataLoader(
    testset, batch_size=args().batch_size, shuffle=True, **args().kwargs
)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
