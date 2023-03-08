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

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
import cv2


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


def plot_train_test_acc_loss(train_losses, train_acc, test_losses, test_acc):
    t = [t.cpu().item() for t in train_losses]
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(t)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    
    
def plot_misclassified_images(incorr_X, incorr_y, incorr_argmax, classes):
    fig = plt.figure(figsize=(20, 8))
    fig.suptitle("CIFAR Mis-Classified Images", fontsize=15)
    for idx in range(10):
        ax = fig.add_subplot(2, 5, idx+1, xticks=[], yticks=[])
        img = np.squeeze(incorr_X[idx])
        img = img.permute(1, 2, 0)
        label = incorr_y[idx]
        img_pred = incorr_argmax[idx]
        plt.imshow(img.cpu().numpy(), cmap="gray")
        ax.set_title(f"Prediction : {classes[img_pred]} \n (label: {classes[label]})")

def plot_gradcam(model, images, labels, predictions, classes ,layer, device, num_of_images=10):
    fig = plt.figure(figsize=(32, 32))
    nrow = num_of_images
    ncol = 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(32,32), squeeze=False)
    
    for idx in range(num_of_images):
        '''
        images - list of images to be processed
        labels - what is the ground truth for the image
        predictions - what the model has been predicting
        layer - GradCAM for the required layer
        '''
        # Given an image, we denormalize it. 
        # As this is CIFAR dataset, the below means and stds are for this specific data set
        input_tensor = images[idx]
        input_tensor = torch.tensor(input_tensor)
        denorms = denormalize(input_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # We unsqueeze to add the batch parameter to the tensor
        input_tensor = torch.unsqueeze(denorms, 0)
        input_tensor = input_tensor.permute(0,3, 2, 1) 
        # This step is to permute the data in the required input format
        # We move the channels from end to second input - 3 --> 0
        input_tensor = input_tensor.to(device)
        img = input_tensor
        
        targets = [ClassifierOutputTarget(labels[idx])]
        target_layers = layer
        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
            img = torch.squeeze(img,0)
            img = img.permute(2,1,0)
            img = img.cpu().numpy()
            cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
        cam = np.uint8(255*grayscale_cams[0, :])
        cam = cv2.merge([cam, cam, cam])
        img = cv2.resize(img, (224, 224))
        cam = cv2.resize(cam, (224, 224))
        cam_image = cv2.resize(cam_image, (224, 224))

        label = labels[idx]
        img_pred = predictions[idx]
        
        # Show the Original Image
        ax = axes[idx, 0]
        ax.imshow(img)
        ax.set_title(f"Original Image: {classes[label]}")
        ax.axis('off')
        
        ax = axes[idx, 1]
        ax.imshow(cam)
        ax.set_title(f"GradCAM Plot for Prediction: {classes[img_pred]})")
        ax.axis('off')
        
        ax = axes[idx, 2]
        ax.imshow(cam_image)
        ax.set_title(f"Prediction : {classes[img_pred]} \n (label: {classes[label]})")
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

def denormalize(x, mean, std):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute( 0, 1, 2)
    
