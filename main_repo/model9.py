import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
dropout_value = 0.0

class UltimusBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes = 48, planes = 8):
        super(UltimusBlock, self).__init__()
        
        self.dimensions = planes
        self.sqrt_dimensions = self.dimensions ** 0.5
        
        self.key = nn.Linear(in_planes, planes)
        self.query = nn.Linear(in_planes, planes)
        self.value = nn.Linear(in_planes, planes)

        self.out = nn.Linear(planes, in_planes)

    def forward(self, x):
        key = self.key(x)
        query = self.query(x)
        value = self.query(x)
        query_t = torch.transpose(query, 0, 1)
        step1 = torch.matmul(query_t,key)
        step2 = step1/(self.sqrt_dimensions)
        softmax_initiation = nn.Softmax(dim = 1)
        AM = softmax_initiation(step2)
        z = torch.matmul(value, AM)
        out = self.out(z)
        return out

class Net(nn.Module):
    def __init__(self, dropout_value=0.0, num_classes=10):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),  
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # Input: 32x32x3 | Output: 32x32x16 | RF: 3
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),  # Input: 32x32x16 | Output: 32x32x32 | RF: 5
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), padding=1, bias=False),  # Input: 32x32x32 | Output: 32x32x48 | RF: 7
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )
        self.gap1 = nn.MaxPool2d(kernel_size=32,stride = 1)
        self.UltimusBlock = UltimusBlock(in_planes=48, planes=8)
        self.final_FC_layer = nn.Linear(48, 10)
    
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.gap1(x)
        x = x.view(-1, 48)
        x = self.UltimusBlock(x)
        x = self.UltimusBlock(x)
        x = self.UltimusBlock(x)
        x = self.UltimusBlock(x)
        x = self.final_FC_layer(x)
        return F.log_softmax(x, dim=-1)
