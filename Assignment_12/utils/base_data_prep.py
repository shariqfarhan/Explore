#Custom code to prepare data for YOLO detection, preparing the base files

import os
import glob
path = os.getcwd()
filenames = glob.glob(path+'/'+'*jpeg')
filenames.sort()
len(filenames)
# 110


# Custom code to get Image Height & Width

with open('custom.shapes','w+') as f:
     for i in range(len(filenames)):
             imageread = cv2.imread(filenames[i], cv2.IMREAD_UNCHANGED)
             height = imageread.shape[0]
             width = imageread.shape[1]
             f.write(str(height) + ' ' + str( width) + '\n')
     f.close()

with open('custom.txt','w+') as f:
     for i in range(len(filenames)):
             f.write(filenames[i]+'\n')
     f.close()

# Rename Files and save it to a dictionary
import shutil
from shutil import copy
path = os.getcwd()
new_path = os.path.dirname(path)
image_names_dict = {}
old_names, new_names = list(), list()
length_of_images = len(filenames)
for i in range(length_of_images):
    n = 3
    pad = '0'
    x = str(i+1).rjust(n, pad)
    my_dest ="img-" + str(x) + ".jpeg"
    my_source = filenames[i]
    my_dest = new_path + '/' +  'Test' +'/' + my_dest
    shutil.copy(my_source, my_dest)
    # os.rename(my_source, my_dest)
    image_names_dict[my_source] = my_dest
    old_names.append(my_source)
    new_names.append(my_dest)

with open('image_names_dict.txt', 'w') as f:
    f.write(str(image_names_dict))
    f.close()


# Code to read file names from dictionary - Mapping old file names to new
import os
import glob
path = os.getcwd()
filenames = glob.glob(path+'/'+'*txt')
filenames.sort()
len(filenames)

import json
with open('image_names_dict.txt') as f:
    data = f.read()

js = json.loads(data)

import re
path = os.getcwd()
new_path = os.path.dirname(path)
length_of_images = len(filenames)
for i in range(length_of_images):
    head, tail = os.path.split(filenames[i])
    my_source = filenames[i]
    file_key = tail.replace(".txt","")
    label_name = js[file_key]+".txt"
    my_dest = new_path + '/' +  'labels' +'/' + label_name
    shutil.copy(my_source, my_dest)
