import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
dropout_value = 0.0

class UltimusBlock(nn.Module):
    expansion = 1

    def __init__(self, dim_inp=48, dim_out=8, bias = False):  
        super(UltimusBlock, self).__init__()  
  
        self.dim_inp = dim_inp
        self.dim_out = dim_out
        
  
        self.q = nn.Linear(dim_inp, dim_out)  
        self.k = nn.Linear(dim_inp, dim_out)  
        self.v = nn.Linear(dim_inp, dim_out)
        self.out = nn.Linear(dim_out, dim_inp)
    
    def forward(self, input_tensor: torch.Tensor):  
        query, key, value = self.q(input_tensor), self.k(input_tensor), self.v(input_tensor)
        key = key.view(key.size(0), 1, key.size(-1))
        query = query.view(query.size(0), 1, query.size(-1))
        value = value.view(value.size(0), 1, value.size(-1))
  
        scale = self.dim_out ** 0.5
        query_t = torch.transpose(query, 1, 2)
        AM = torch.matmul(query_t, key) / scale
        AM = F.softmax(AM, dim = -1)
        Z = torch.matmul(value, AM)
        out_1 = self.out(Z)
        out_1 = out_1.view(-1, out_1.size(-1))
        return out_1

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
        self.UltimusBlock = UltimusBlock(dim_inp=48, dim_out=8, bias = False)
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
