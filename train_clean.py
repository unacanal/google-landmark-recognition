import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

import pandas as pd
import os

from PIL import Image

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

EPOCHS = 40
BATCH_SIZE = 64

# load train data (landmark_id, images)
base_dir = '/mnt/Datasets/GLDv2'
train_path = os.path.join(base_dir, 'train/train_clean.csv')
# train_path = os.path.join(base_dir, 'train/train.csv')
train_data = pd.read_csv(train_path)

# print(train_data.head()) # 81313 rows
# print(train_data.groupby('landmark_id').count()) # What did I mean to do?
#
for index, row in train_data.iterrows():
    y = row['landmark_id']
    x = row['images']
    print(y, x)
    break

x_list = x.split(' ')
for x in x_list:
    # folder name == each letter from first 3 letters of image file name
    img_path = os.path.join(base_dir, 'train', x[0], x[1], x[2], x + '.jpg')

    image = Image.open(img_path)
    image.show()
