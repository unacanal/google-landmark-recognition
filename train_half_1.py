import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import os
from PIL import Image
from skimage import io

import time

from torch.utils.tensorboard import SummaryWriter

class GLDTrain(Dataset):
    def __init__(self,  transform=None):
        # self.annotations = pd.read_csv('/mnt/Datasets/GLDv2/train/train_clean_clean_less3.csv')
        self.annotations = pd.read_csv('train_clean_clean_less3_half_1.csv')
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        base_dir = '/mnt/Datasets/GLDv2'
        test_path = os.path.join(base_dir, 'train')
        test_data = self.annotations
        img_file = test_data.iloc[index, 0] + '.jpg'
        img_path = os.path.join(test_path, img_file[0], img_file[1], img_file[2], img_file)

        try:
            image = io.imread(img_path)
            image = Image.fromarray(image)
            y_label = torch.tensor(int(test_data.iloc[index, 1]) - 1)

            if self.transform:
                image = self.transform(image)

            return (image, y_label)
        except FileNotFoundError as e:
            print(e)


class GLDTest(Dataset):
    def __init__(self,  transform=None):
        self.annotations = pd.read_csv('test_clean_half_1.csv')
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        base_dir = '/mnt/Datasets/GLDv2'
        test_path = os.path.join(base_dir, 'test')
        test_data = self.annotations
        img_file = test_data.iloc[index, 0] + '.jpg'
        img_path = os.path.join(test_path, img_file[0], img_file[1], img_file[2], img_file)

        image = io.imread(img_path)
        image = Image.fromarray(image)
        y_label = torch.tensor(int(test_data.iloc[index, 1]) - 1)
        # print(img_path)

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

NUM_CLASSES = 101547 # 203094 # original data
BATCH_SIZE = 32
EPOCHS = 32 * 32

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

train_data = GLDTrain(transform=transform)
print(len(train_data))
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = GLDTest(transform=transform)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

# def pickle_loader(input):
#     item = pickle.load(open(input, 'rb'))
#     return item.values
# train_data = datasets.DatasetFolder(root='.', loader=pickle_loader, extensions='.pkl', transform=transform)
# train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
#
# test_data = datasets.DatasetFolder(root='.', loader=pickle_loader, extensions='.pkl', transform=transform)
# test_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)

model = models.vgg19(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
# VGG
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
model.num_classes = NUM_CLASSES
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier[6].parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

writer = SummaryWriter(comment=model.__class__.__name__)

timestr = time.strftime("%Y%m%d-%H%M%S")
MODEL_PATH = os.path.join('models', model.__class__.__name__, timestr)

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()

    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        pred = model(data)
        # print(torch.utils.checkpoint.checkpoint(model.forward()))
        loss = criterion(pred, target)
        del pred
        loss.backward()
        optimizer.step()

        running_loss += loss.item() #data[0]
        # if batch_idx % 200 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'
        #           .format(epoch, batch_idx * len(data),
        #                   len(train_loader.dataset),
        #                   100. * batch_idx / len(train_loader),
        #                   loss.data[0]))
        print("==> Iteration:", batch_idx)
        if batch_idx % 1000 == 999:
            writer.add_scalar('training_loss', running_loss / 1000,
                              epoch * len(train_loader) + batch_idx)
            running_loss = 0.0

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    writer.add_scalar('test_loss', test_loss)
    writer.add_scalar('test_accruacy', test_accuracy)

for epoch in range(EPOCHS):
    print("====> EPOCH:", epoch)
    train(model, train_loader, criterion, optimizer, epoch)

    lr_scheduler.step()

    if epoch % 10 == 0:
         evaluate(model, test_loader)

torch.save(model, MODEL_PATH)