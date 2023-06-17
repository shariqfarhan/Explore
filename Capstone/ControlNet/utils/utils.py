import cv2
import numpy as np
import torch
from torchvision import transforms


def convert_tensor_to_canny(image_list):
    canny_image_list = []
    for i in range(len(image_list)):
        transform = T.ToPILImage()
        x = image_list[i]
        img = transform(x.squeeze())
        image = np.array(img)
        image = np.uint8(image)

        low_threshold = 100
        high_threshold = 200

        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)
        canny_image_list.append(canny_image)
    return canny_image_list
