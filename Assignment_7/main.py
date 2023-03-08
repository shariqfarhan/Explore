
#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(u'git clone https://github.com/shariqfarhan/EVA8-pytorch-cifar')


# In[2]:


import os
os.chdir('EVA8-pytorch-cifar')
os.listdir()


# In[7]:


get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from torchinfo import summary
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt



from models import resnet
from utils.utils import get_mean_and_std, show_data_metrics, viz_data, show_images, show_model_summary
from utils.utils import AlbumentationImageDataset, plot_train_test_acc_loss, plot_misclassified_images, denormalize, plot_gradcam
from train import train
from test_model import test, mis_classified_images


# In[5]:


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

exp = datasets.CIFAR10('./data', train = True, download = True)
exp_data = exp.data

show_data_metrics(exp_data)


# In[6]:


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[8]:


viz_data(exp)


# In[9]:


# RandomCrop(32, padding=4)
# CutOut(16x16)
show_images(exp,{
    'Original Image' : None,
    'Additional Padding' : A.PadIfNeeded(min_height=40, min_width=40), # Padding on both sides equal to 4 -> 32 + 2*4 = 40
    'Random Crop' : A.RandomCrop(32, 32),
    'Cut Out' : A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16,
                                min_width=16, fill_value=0.473363, mask_fill_value=None, always_apply=True)
}, ncol = 4)


# In[10]:


# session_parameters()
SEED = 1

# CUDA
cuda = torch.cuda.is_available()
print("CUDA Available:", cuda)

# For Reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)
    BATCH_SIZE = 64
else:
    BATCH_SIZE = 64


# In[11]:


trainset = torchvision.datasets.CIFAR10(root='./data', train = True, 
                                        download = True)
testset = torchvision.datasets.CIFAR10(root='./data', train = False, 
                                        download = True)

train_loader = torch.utils.data.DataLoader(AlbumentationImageDataset(trainset, train = True),
                                          batch_size = BATCH_SIZE, shuffle=True,num_workers = 2)
test_loader = torch.utils.data.DataLoader(AlbumentationImageDataset(testset, train = False),
                                          batch_size = BATCH_SIZE, shuffle=True,num_workers = 2)


# In[12]:


from models import *
from models.resnet import ResNet18
model = ResNet18()
show_model_summary(model)


# In[13]:


import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
EPOCHS = 20

model = ResNet18().to(device)


# In[14]:


optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()
train_losses, train_acc, test_losses, test_acc = list(), list(), list(), list()
for epoch in range(1, EPOCHS+1):
    print("EPOCH: {} (LR: {})".format(epoch, optimizer.param_groups[0]['lr']))
    train(model, device, train_loader, optimizer, criterion, train_losses, train_acc, l1_penalty = True, lambda_l1 = 1e-5)
    test(model, device, test_loader, test_losses, test_acc)


# In[15]:


plot_train_test_acc_loss(train_losses, train_acc, test_losses, test_acc)


# In[16]:




# Identify incorrect classifications
incorr_X, incorr_y, incorr_argmax = mis_classified_images(model, device, test_loader) 

#Plot the images
plot_misclassified_images(incorr_X, incorr_y, incorr_argmax, classes)


# In[17]:


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
import cv2


# In[25]:


layer = [model.layer1[-2]]
plot_gradcam(model, incorr_X, incorr_y, incorr_argmax, classes ,layer, device, 10)


# In[19]:


print('Total Model Parameters : ', sum(p.numel() for p in model.parameters()))
print('Best Train Accuracy : ', max(train_acc))
print('Best Test Accuracy : ', max(test_acc))
x = [x for x in test_acc if x >= 85]
print('# of Epochs with test accuracy above 85% threshold:', len(x))



