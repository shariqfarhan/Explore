# Calculate the mean and the std for normalization

import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from torchinfo import summary

import numpy as np
import matplotlib.pyplot as plt

import albumentations as A
from tqdm import tqdm


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def show_data_metrics(data):
    # This function is used to derive various metrics to understand about the data
    '''
    Data - In this case is CIFAR10, but this function could be used for any dataset
    '''
    exp_data = data
    print('Train')
    print('- Numpy Shape :', exp_data.shape)
    print('- min :', np.min(exp_data, axis = (0,1,2)) / 255.)
    print('- max :', np.max(exp_data, axis = (0,1,2)) / 255.)
    print('- mean :', np.mean(exp_data, axis = (0,1,2)) / 255.)
    print('- std. :', np.std(exp_data, axis = (0,1,2)) / 255.)
    print('- var :', np.var(exp_data, axis = (0,1,2)) / 255.)
    
    

def viz_data(exp, cols=8, rows=5):
    '''
    This function is used to show a few sample images of the CIFAR dataset
    exp is the input data through which this function is fed.
    '''
    figure = plt.figure(figsize = (14,10))
    for i in range(1, cols * rows + 1):
        img, label = exp[i]
        
        figure.add_subplot(rows, cols, i)
        plt.title(exp.classes[label])
        plt.axis('off')
        plt.imshow(img, cmap = 'gray')
    
    plt.tight_layout()
    plt.show()

    
def show_images(exp,aug_dict, ncol = 6):
    '''
    This function is used to show the original image and the associated image after augmentation.
    '''
    nrow = len(aug_dict)
    
    fig, axes = plt.subplots(ncol, nrow, figsize=(3*nrow, 15), squeeze=False)
    for i, (key, aug) in enumerate(aug_dict.items()):
        for j in range(ncol):
            ax = axes[j, i]
            if j == 0:
                ax.text(0.5, 0.5, key, horizontalalignment='center', verticalalignment='center', fontsize=15)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.axis('off')
            else:
                image, label = exp[j-1]
                if aug is not None:
                    transform = A.Compose([aug])
                    image = np.array(image)
                    image = transform(image=image)['image']
                ax.imshow(image)
                ax.set_title(f'{exp.classes[label]}')
                ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def show_model_summary(model):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    net = model.to(device)
    batch_size = 16
    print(summary(net, input_size=(batch_size,3,32,32)))

def session_parameters():
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
    
class AlbumentationImageDataset(Dataset):
    def __init__(self, image_list, train = True):
        self.image_list = image_list
        self.aug = A.Compose({
            A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
            A.HorizontalFlip(),
            A.ShiftScaleRotate(),
            A.CoarseDropout(1,16,16,1,16,16,fill_value = 0.473363, mask_fill_value= None),
            A.ToGray()
        })
        self.norm = A.Compose({A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))})
        self.train = train
        
    def __len__(self):
        return (len(self.image_list))
    
    def __getitem__(self, i):
        image, label = self.image_list[i]
        
        if self.train:
            image = self.aug(image = np.array(image))['image']
        else:
            image = self.norm(image = np.array(image))['image']
        image = np.transpose(image, (2,0,1)).astype(np.float32)
        return torch.tensor(image, dtype = torch.float), label    
    
    
