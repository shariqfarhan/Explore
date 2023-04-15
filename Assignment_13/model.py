import torch
import torch.nn as nn
import torchvision.transforms.functional


class EncoderMiniBlock(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=32, dropout=0.3, max_pooling=True):
        super(EncoderMiniBlock, self).__init__()
        
        self.convblock1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels = out_channels, kernel_size=(3, 3), padding=1, bias=False),  # Input: 256x256x32 | Output: 256x256x32 | RF: 3x3
                nn.ReLU(),
                nn.Conv2d(in_channels=out_channels, out_channels = out_channels, kernel_size=(3, 3), padding=1, bias=False),  # Input: 256x256x32 | Output: 256x256x32 | RF: 5x5
                nn.ReLU(),
            # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
                nn.BatchNorm2d(out_channels),
            # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
                nn.Dropout(dropout),
            )
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride = 2)
        self.max_pooling = max_pooling
        
    def forward(self, x):
        out = self.convblock1(x)
        
        # Pooling reduces the size of the image while keeping the number of channels same
        # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use)
        # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse across input image
        if self.max_pooling:
            next_layer = self.max_pool(out)
        else:
            next_layer = out
        
        # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions      
        skip_connection = out
        return next_layer, skip_connection # Out is the skip Connection

class DecoderMiniBlock(nn.Module):
    
    def __init__(self, in_channels=32, out_channels=32):
        super(DecoderMiniBlock, self).__init__()
        
        
        self.deconvblock1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels = out_channels, kernel_size=(2, 2), stride = 2 , bias=False)
        )
        
        # Add 2 Conv Layers with relu activation using nn
        # padding = 1 will pad the input to conv layer such that the output has the same height and width (hence, is not reduced in size)         
        
        self.convblock1 = nn.Sequential(
                nn.Conv2d(in_channels=out_channels*2, out_channels = out_channels, kernel_size=(3, 3), padding=1, bias=False),  # Input: 256x256x32 | Output: 256x256x32 | RF: 3x3
                nn.ReLU(),
                nn.Conv2d(in_channels=out_channels, out_channels = out_channels, kernel_size=(3, 3), padding=1, bias=False),  # Input: 256x256x32 | Output: 256x256x32 | RF: 5x5
                nn.ReLU(),
            # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
                nn.BatchNorm2d(out_channels),
            # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
            )
        
    def forward(self, x, y): # x - previous layer input, y - skip connection output from Encoder
        up = self.deconvblock1(x)
#         print(f"Shape after deconv {up.shape}")
#         print(f"Shape of 2nd input {y.shape}")
        merge = torch.cat([up, y], dim = 1)
#         print(f"Shape of Merge {merge.shape}")
        out = self.convblock1(merge)
        
        return out

class DoubleConvolution(nn.Module):
    """
    ### Two $3 \times 3$ Convolution Layers
    Each step in the contraction path and expansive path have two $3 \times 3$
    convolutional layers followed by ReLU activations.
    In the U-Net paper they used $0$ padding,
    but we use $1$ padding so that final feature map is not cropped.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: is the number of input channels
        :param out_channels: is the number of output channels
        """
        super().__init__()

        # First $3 \times 3$ convolutional layer
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        # Second $3 \times 3$ convolutional layer
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # Apply the two convolution layers and activations
        x = self.first(x)
        x = self.act1(x)
        x = self.second(x)
        return self.act2(x)

class UNet(nn.Module):
    def __init__(self, n_filters=32, n_classes=3):
        super(UNet, self).__init__()
        
        # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
        # Observe that the channels are increasing as we go deeper into the network which will increasse the # channels of the image 
        
        self.cblock1 = EncoderMiniBlock(in_channels = 3,out_channels = n_filters*1,dropout=0, max_pooling=True)
        self.cblock2 = EncoderMiniBlock(in_channels = n_filters*1,out_channels = n_filters*2,dropout=0, max_pooling=True)
        self.cblock3 = EncoderMiniBlock(in_channels = n_filters*2,out_channels = n_filters*4,dropout=0, max_pooling=True)
        self.cblock4 = EncoderMiniBlock(in_channels = n_filters*4,out_channels = n_filters*8,dropout=0.3, max_pooling=True)
        self.cblock5 = EncoderMiniBlock(in_channels = n_filters*8,out_channels = n_filters*16,dropout=0.3, max_pooling=True)
        
        self.cblock6 = EncoderMiniBlock(in_channels = n_filters*16,out_channels = n_filters*32,dropout=0.3, max_pooling=False)
        
        # Decoder includes multiple mini blocks with decreasing number of filters
        # Observe the skip connections from the encoder are given as input to the decoder
        # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
        
        self.ublock6 = DecoderMiniBlock(in_channels = n_filters*32,out_channels = n_filters*16)
        self.ublock7 = DecoderMiniBlock(in_channels = n_filters*16,out_channels = n_filters*8)
        self.ublock8 = DecoderMiniBlock(in_channels = n_filters*8,out_channels = n_filters*4)
        self.ublock9 = DecoderMiniBlock(in_channels = n_filters*4,out_channels = n_filters*2)
        self.ublock10 = DecoderMiniBlock(in_channels = n_filters*2,out_channels = n_filters*1)
        
        self.conv1 = DoubleConvolution(in_channels = n_filters,out_channels = n_classes)
    
    def forward(self, x):
        out1 = self.cblock1(x)
        out2 = self.cblock2(out1[0])
        out3 = self.cblock3(out2[0])
        out4 = self.cblock4(out3[0])
        out5 = self.cblock5(out4[0])
        out6 = self.cblock6(out5[0])
        
        out7 = self.ublock6(out6[0], out5[1])
        out8 = self.ublock7(out7, out4[1])
        out9 = self.ublock8(out8, out3[1])
        out10 = self.ublock9(out9, out2[1])
        out11 = self.ublock10(out10, out1[1])
        
        
        out12 = self.conv1(out11)
        return out12
