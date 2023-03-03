# Custom ResNet

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, drop=0.025, num_classes=10):
        super(Net, self).__init__()

        self.prep_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ) # Input: 32x32x3 | Output: 32x32x64 | RF: 3x3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),  # Input: 32x32x64 | Output: 32x32x128 | RF: 5x5
            nn.MaxPool2d(kernel_size=2, stride=2), # Input: 32x32x128 | Output: 16x16x128 | RF: 6x6
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.ResBlock1 = ResBlock(64, 128, 2) # Input: 32x32x64 | Output: 16x16x128 | RF: 14x14

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),  # Input: 16x16x128 | Output: 16x16x256 | RF: 18x18
            nn.MaxPool2d(kernel_size=2, stride=2), # Input: 16x16x256 | Output: 8x8x256 | RF: 20x20            
            nn.BatchNorm2d(256),
            nn.ReLU()
        )# Input: 16x16x128 | Output: 8x8x256 | RF: 20x20
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),  # Input: 8x8x256 | Output: 8x8x512 | RF: 28
            nn.MaxPool2d(kernel_size=2, stride=2), # Input: 8x8x512 | Output: 4x4x512 | RF: 32            
            nn.BatchNorm2d(512),
            nn.ReLU()
        )# Input: 8x8x256 | Output: 4x4x512 | RF: 20x20
        self.ResBlock2 = ResBlock(256, 512, 2) # Input: 8x8x256 | Output: 4x4x512 | RF: 64
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512) 
        )# Input: 16x16x128 | Output: 4x4x512 | RF: 64
        self.final_pooling = nn.MaxPool2d(kernel_size=4,stride = 1) # Input: 4x4x512 | Output: 1x1x512 | RF: 88
        self.fc1 = nn.Linear(512,num_classes)
    
    def forward(self, x):     
        residual = x.clone()
        out = self.prep_layer1(x)
        out1 = self.convblock1(out)
        out2 = self.ResBlock1(out)
        out = out1 + out2 # Adding Outputs from Convolution block & ResNet Block
        out = self.convblock2(out)
        out1 = self.convblock3(out)
        out2 = self.ResBlock2(out)
        out = out1 + out2 # Adding Outputs from Convolution block & ResNet Block
        out = self.final_pooling(out)
        out = out.view(-1, 512)                    
        out = self.fc1(out)
        return F.log_softmax(out, dim=-1)
