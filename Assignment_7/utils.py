import numpy as np
import torch
import matplotlib.pyplot as plt
from torchsummary import summary


'''
The functions in this file can be used to 

1. understand the data
2. transformations on the data
3. Data Augmentation
4. Understand Predictions using GradCAM
5. Showing Mis-Classified Images

'''
# Function to understand the basic metrics about the data

def show_data_metrics(data):
    # This function is used to derive various metrics to understand about the data
    exp_data = data
    print('Train')
    print('- Numpy Shape :', exp_data.shape)
    print('- min :', np.min(exp_data, axis = (0,1,2)) / 255.)
    print('- max :', np.max(exp_data, axis = (0,1,2)) / 255.)
    print('- mean :', np.mean(exp_data, axis = (0,1,2)) / 255.)
    print('- std. :', np.std(exp_data, axis = (0,1,2)) / 255.)
    print('- var :', np.var(exp_data, axis = (0,1,2)) / 255.)


# Function to show sample images on the dataset

def sample_data(cols=8, rows=5):
    figure = plt.figure(figsize = (14,10))
    for i in range(1, cols * rows + 1):
        img, label = exp[i]
        
        figure.add_subplot(rows, cols, i)
        plt.title(exp.classes[label])
        plt.axis('off')
        plt.imshow(img, cmap = 'gray')
    
    plt.tight_layout()
    plt.show()

# Function to show base images and images after applying augmentations
    
def show_images(aug_dict, ncol = 6):
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
 
# Function to show Model Summary
    
def show_model_summary(model):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    net = model(drop=0.0).to(device)
    summary(net, input_size=(3,32,32))

# Class function to generate an augmented dataset given an input set of images from CIFAR10 dataset
# To apply this to another dataset, we need to change the inputs to the normalize functions - as each dataset would have it's own set of parameters
    
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
