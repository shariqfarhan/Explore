import torch.nn as nn
import torch.nn.functional as F
dropout_value = 0.0

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convblock_0 = nn.Sequential(
                       nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3),dilation=1,stride=1,padding=1,bias=False,),
                       nn.ReLU(),
                       nn.BatchNorm2d(16),
                       nn.Dropout(dropout_value), # Input - 32x32x3 | Output - 32X32X16 | RF=3

                       nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),dilation=1,stride=1,padding=1,bias=False,),
                       nn.ReLU(),
                       nn.BatchNorm2d(32),
                       nn.Dropout(dropout_value), # Input - 32X32X16 | Output - 32X32x32 |RF=5

                       nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),dilation=1,stride=1,padding=1,bias=False,),
                       nn.ReLU(),
                       nn.BatchNorm2d(32),
                       nn.Dropout(dropout_value), # Input - 32X32X32 | Output - 32X32X64 |RF= 7

                       nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),dilation=1,stride=1,padding=1,bias=False,),
                       nn.ReLU(),
                       nn.BatchNorm2d(32),
                       nn.Dropout(dropout_value), # Input - 32X32X64 | Output - 32X32X64 |RF= 9
                      )
        
        # depthwise seperable Convolution 1
        self.convblock_1 = nn.Sequential(
        
                       nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=(2,2),dilation=1,padding=1,bias=False,),# maxpool added after RF >11
                       nn.ReLU(),
                       nn.BatchNorm2d(64),
                       nn.Dropout(dropout_value), # Input - 32X32X64 | Output - 16X16X64 |RF=11

                       nn.Conv2d(in_channels=64,out_channels=64,groups=64,kernel_size=(3,3),stride=(1,1),dilation=1,padding=1,bias=False,),
                       # Input - 16X16X64 | Output - 16X16X64 | RF=15
                       nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(1,1),stride=(1,1),padding=0,bias=False,),
                       # Input - 16X16X64 | Output - 16X16X64 | RF=15
                       nn.ReLU(),
                       nn.BatchNorm2d(128), 
                       nn.Dropout(dropout_value), # 16X16X64 | RF=21                                       
                       # pointwise   

                       nn.Conv2d(in_channels=128,out_channels=128,groups=128,kernel_size=(3,3),dilation=1,stride=(1,1),padding=1,bias=False,),
                       # Input - 16X16X64 | Output - 16X16X64 | RF=29
                       nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(1,1),padding=0,bias=False,),
                       nn.ReLU(),
                       nn.BatchNorm2d(64),   
                       nn.Dropout(dropout_value), 
                       # Input - 16X16X64 | Output - 16X16X32 | RF=29
                       
                      #nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(1,1),dilation=2,padding=1,bias=False,),
                      # #  nn.Conv2d(in_channels=64,out_channels=64,groups=64,kernel_size=(3,3),stride=(1,1),padding=1,bias=False,),
                      # #  nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(1,1),padding=0,bias=False,),
                      #  nn.ReLU(),
                      #  nn.BatchNorm2d(64),   
                      #  nn.Dropout(dropout_value) , # 16X16X64 | RF=29                                                         
                       )
        # depthwise seperable Convolution 2
        self.convblock_2 = nn.Sequential(
        
                       nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3),stride=(2,2),dilation=1,padding=1,bias=False,),
                       nn.ReLU(),
                       nn.BatchNorm2d(32),   
                       nn.Dropout(dropout_value), 
                      # # Input - 16X16X32 | Output - 8X8X32 | RF=37

                       nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),stride=(1,1),padding=1,bias=False,),
                       # # Input - 8X8X32 | Output - 8X8X32 | RF=45
                       nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(1,1),stride=(1,1),padding=0,bias=False,),
                       # # Input - 8X8X32 | Output - 8X8X64 | RF=45
                       nn.ReLU(),
                       nn.BatchNorm2d(64),  
                       nn.Dropout(dropout_value),
                      # pointwise   

                       nn.Conv2d(in_channels=64,out_channels=64,groups=64,kernel_size=(3,3),stride=(1,1),padding=1,bias=False,),
                       # # Input - 8X8X64 | Output - 8X8X128 | RF=53
                       nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(1,1),stride=(1,1),padding=0,bias=False,),
                       nn.ReLU(),
                       nn.BatchNorm2d(64),  # pointwise 
                       nn.Dropout(dropout_value) 
                       # # Input - 8X8X64 | Output - 8X8X64 | RF=53

                      )
        # depthwise seperable Convolution 2
        self.convblock_3 = nn.Sequential(
        
                       #Maxpooling
                       nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),dilation=1,stride=(2,2),padding=1,bias=False),
                       nn.ReLU(),
                       nn.BatchNorm2d(64),  
                       nn.Dropout(dropout_value),
                      # # Input - 8X8X64 | Output - 4X4X64 | RF=69

                       nn.Conv2d(in_channels=64,out_channels=128,groups=64,kernel_size=(3,3),stride=(1,1),padding=1,bias=False,),
                       # # Input - 4X4X64 | Output - 4X4X128 | RF=85
                       nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(1,1),stride=(1,1),padding=0,bias=False,),
                       nn.ReLU(),
                       nn.BatchNorm2d(64),
                       nn.Dropout(dropout_value),
                      #  # Input - 4X4X128 | Output - 4X4X64 | RF=85
                     
                       nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(3,3),dilation=2,stride=(1,1),padding=2,bias=False),
                       nn.ReLU(),
                       nn.BatchNorm2d(32),  
                       nn.Dropout(dropout_value),  
                       #  # Input - 4X4X128 | Output - 4X4X64 | RF=117

                       nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
                       # Input - 4X4X32 | Output - 4X4X10 | RF=117

                       )
        # 4X4X10 | RF=121
        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=4))
        self.fc = nn.Linear(in_features = 10, out_features = 10)

        # Input - 4X4X64 | Output - 1X1X64 
        # self.fc1 = nn.Linear(128,10)
        # self.fc2 = nn.Linear(64,10)

    
    def forward(self, x):
      
      x = self.convblock_0(x)
      x = self.convblock_1(x)
      x = self.convblock_2(x)
      x = self.convblock_3(x)
      x = self.gap(x)
      x = x.view(x.size(0), -1)
      x = self.fc(x)
      # x = self.fc1(x)
      # x = self.fc2(x)     
      x = x.view(-1, 10)
      return F.log_softmax(x, dim=1)

net = Net()
