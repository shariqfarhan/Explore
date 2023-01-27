# We change the way we define the neural network. Earlier we used to work one convolution layer at a time, now we'll add more operations on top of the convolution layer
# All of this will fall under 1 block.
dropout_value = 0.1

class Net(nn.Module):
    
    def __init__(self, use_ln: bool = False, use_BN: bool = True, use_GN:bool = False):
        
        super(Net, self).__init__()
        
        # Determining the Type of Normalization
        self.use_ln = use_ln # Initializing the Layer Normalization variable - If set to True, Layer Normalization is True
        self.use_BN = use_BN # Initializing the Batch Normalization variable - If set to True, Batch Normalization is True
        self.use_GN = use_GN # Initializing the Group Normalization variable - If set to True, Group Normalization is True
        
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(4)  if self.use_BN else nn.Identity(),
            nn.ReLU(),
            nn.LayerNorm(26, elementwise_affine = False) if self.use_ln else nn.Identity(), # 26 is the Output from the 1st layer of Conv
            nn.GroupNorm(2, 4) if self.use_GN else nn.Identity() # We split 4 channels into 2 Groups
        ) # Input - 28 x 28 x 1 | Output - 26 x 26 x 4 | RF - 3 | Kernel - 3 x 3 x 1 x 4

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8)  if self.use_BN else nn.Identity(),
            nn.ReLU(),
            nn.LayerNorm(24, elementwise_affine = False) if self.use_ln else nn.Identity() , # 24 is the Output from the 2nd layer of Conv
            nn.GroupNorm(2, 8) if self.use_GN else nn.Identity() # We split 8 channels into 2 Groups
        ) # Input - 26 x 26 x 4 | Output - 24 x 24 x 8 | RF - 5 | Kernel - 3 x 3 x 4 x 8
        
        # TRANSITION BLOCK 1
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        # Input - 24 x 24 x 8 | Output - 12 x 12 x 8 | RF - 10 | Kernel - 2 x 2

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16)   if self.use_BN else nn.Identity(),
            nn.ReLU(),
            nn.LayerNorm(10, elementwise_affine = False) if self.use_ln else nn.Identity(), # 10 is the Output from the 1st layer of Conv
            nn.GroupNorm(2, 16) if self.use_GN else nn.Identity(), # We split 16 channels into 2 Groups
            nn.Dropout(dropout_value)
        ) # Input - 12 x 12 x 8 | Output - 10 x 10 x 16 | RF - 12 | Kernel - 3 x 3 x 8 x 16
        
        # TRANSITION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            #nn.BatchNorm2d(16),
            #nn.ReLU(),
            #nn.Dropout(dropout_value)
          # If we increase the number of channels to 32, it increases the number of parameters by 4.6K, hence maintaining the channels at 16 
          # What is the benefit of having this block then? 
        ) # Input - 10 x 10 x 16 | Output - 10 x 10 x 16 | RF - 12 | Kernel - 3 x 3 x 16 x 16

        # CONVOLUTION BLOCK 3
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16)   if self.use_BN else nn.Identity(),
            nn.LayerNorm(8, elementwise_affine = False) if self.use_ln else nn.Identity(),
            nn.GroupNorm(2, 16) if self.use_GN else nn.Identity(), # We split 16 channels into 2 Groups
            nn.ReLU(),
            nn.Dropout(dropout_value)
          # If we increase the number of channels to 32, it increases the number of parameters by 4.6K, hence maintaining the channels at 16 
          # What is the benefit of having this block then? 
        ) # Input - 8 x 8 x 16 | Output - 8 x 8 x 16 | RF - 14 | Kernel - 3 x 3 x 16 x 16

        # CONVOLUTION BLOCK 4
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16)   if self.use_BN else nn.Identity(),
            nn.ReLU(),
            nn.LayerNorm(6, elementwise_affine = False) if self.use_ln else nn.Identity(),
            nn.GroupNorm(2, 16) if self.use_GN else nn.Identity() # We split 16 channels into 2 Groups
        ) # Input - 8 x 8 x 16 | Output - 6 x 6 x 16 | RF - 16 | Kernel - 3 x 3 x 16 x 16
        
        # CONVOLUTION BLOCK 5        
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10)   if self.use_BN else nn.Identity(),
            nn.ReLU(),
            nn.LayerNorm(4, elementwise_affine = False) if self.use_ln else nn.Identity(),
            nn.GroupNorm(2, 10) if self.use_GN else nn.Identity() # We split 10 channels into 2 Groups
        ) # Input - 6 x 6 x 16 | Output - 4 x 4 x 10 | RF - 18 | Kernel - 3 x 3 x 16 x 10

        # GAP Layer
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output_size = 1
        # ) # Input - 6 x 6 x 10 | Output - 1 x 1 x 10 | RF - 21 | Kernel - 6 x 6 x 10 x 10
        
        # Output Layer
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) # ) # Input - 1 x 1 x 10 | Output - 1 x 1 x 10 | RF - 21 | Kernel - 1 x 1 x 10 x 10
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.gap(x)
        x = self.convblock9(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
