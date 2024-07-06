import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
import os
import random
from PIL import Image
import sklearn
from bme1312 import lab2 as lab
from torchvision import transforms as T
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from bme1312.evaluation import get_DC,get_accuracy
from torchvision import transforms


plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

def process_data():
    path="./cine_seg.npz" # input the path of cine_seg.npz in your environment
    dataset=np.load(path,allow_pickle=True)
    files=dataset.files
    inputs=[]
    labels=[]
    for file in files:
        inputs.append(dataset[file][0])
        labels.append(dataset[file][1])
    inputs = torch.Tensor(inputs)
    labels = torch.Tensor(labels)
    return inputs, labels



inputs, labels = process_data()
inputs = inputs.unsqueeze(1)  # Add channel dimension
labels = labels.unsqueeze(1)  # Add channel dimension
print("inputs shape: ", inputs.shape)
print("labels shape: ", labels.shape)


def convert_to_multi_labels(label): 
    device = label.device
    B, C, H, W = label.shape
    new_tensor = torch.zeros((B, 3, H, W), device=device)
    mask1 = (label >= 255).squeeze(1)
    mask2 = ((label >= 170) & (label < 255-1)).squeeze(1)
    mask3 = ((label >= 85) & (label < 170-1)).squeeze(1)
    one = torch.ones(size= (B, H, W), device=device)
    zero = torch.zeros(size=(B, H, W), device=device)
    new_tensor[:, 0, :, :] = torch.where(mask1, one, zero)
    new_tensor[:, 1, :, :] = torch.where(mask2, one, zero)
    new_tensor[:, 2, :, :] = torch.where(mask3, one, zero)
    return new_tensor
    
    
dataset = TensorDataset(inputs, labels) 
batch_size = 32
train_size = int(4/7 * len(dataset))
val_size = int(1/7 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(val_set, batch_size=batch_size, shuffle=True)
dataloader_test =  DataLoader(test_set, batch_size=batch_size, shuffle=True)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, C_base=64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.C_base = C_base

        self.inc = DoubleConv(n_channels, C_base)
        self.down1 = Down(C_base, C_base*2)
        self.down2 = Down(C_base*2, C_base*4)
        self.down3 = Down(C_base*4, C_base*8)
        self.down4 = Down(C_base*8, C_base*8)
        self.up1 = Up(C_base*16, C_base*4, bilinear)
        self.up2 = Up(C_base*8, C_base*2, bilinear)
        self.up3 = Up(C_base*4, C_base, bilinear)
        self.up4 = Up(C_base*2, C_base, bilinear)
        self.outc = nn.Conv2d(C_base, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class MyBinaryCrossEntropy(object):
    def __init__(self):
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss(reduction='mean')

    def __call__(self, pred_seg, seg_gt):
        pred_seg_probs = self.sigmoid(pred_seg)
        seg_gt_probs = convert_to_multi_labels(seg_gt) # convert to multi labels
        loss = self.bce(pred_seg_probs, seg_gt_probs)
        return loss
    
import bme1312.lab2 as lab

net = UNet(n_channels=1, n_classes=3, C_base=32)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

solver = lab.Solver(
    model=net,
    optimizer=optimizer,
    criterion=MyBinaryCrossEntropy(),
    lr_scheduler=lr_scheduler,
)

solver.train(
    epochs=50,
    data_loader=dataloader_train,
    val_loader=dataloader_val,
    save_path='./model_original_unet_bce.pth',
    img_name='original_unet_bce'
)


# calculate the dice coefficient
dice_scores = []
net.to(device)
for images, labels in dataloader_test:
    images = images.to(device)  
    labels = labels.to(device) 
    preds = net(images)
    preds = torch.sigmoid(preds) > 0.5  # Applying threshold to get binary output
    labels = convert_to_multi_labels(labels)  # Convert labels to multi-label format

    dice_rv = get_DC(preds[:, 0, :, :], labels[:, 0, :, :])
    dice_myo = get_DC(preds[:, 1, :, :], labels[:, 1, :, :])
    dice_lv = get_DC(preds[:, 2, :, :], labels[:, 2, :, :])

    dice_scores.append((dice_rv, dice_myo, dice_lv))

mean_dice_rv = np.mean([score[0] for score in dice_scores])
std_dice_rv = np.std([score[0] for score in dice_scores])
mean_dice_myo = np.mean([score[1] for score in dice_scores])
std_dice_myo = np.std([score[1] for score in dice_scores])
mean_dice_lv = np.mean([score[2] for score in dice_scores])
std_dice_lv = np.std([score[2] for score in dice_scores])

print(f'RV Dice Coefficient: Mean={mean_dice_rv}, SD={std_dice_rv}')
print(f'MYO Dice Coefficient: Mean={mean_dice_myo}, SD={std_dice_myo}')
print(f'LV Dice Coefficient: Mean={mean_dice_lv}, SD={std_dice_lv}')



class Up_NoShortcut(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up_NoShortcut, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)


    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)

class UNet_NoShortcut(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, C_base=64):
        super(UNet_NoShortcut, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.C_base = C_base

        self.inc = DoubleConv(n_channels, C_base)
        self.down1 = Down(C_base, C_base*2)
        self.down2 = Down(C_base*2, C_base*4)
        self.down3 = Down(C_base*4, C_base*8)
        self.down4 = Down(C_base*8, C_base*8)
        self.up1 = Up_NoShortcut(C_base*8, C_base*4, bilinear)
        self.up2 = Up_NoShortcut(C_base*4, C_base*2, bilinear)
        self.up3 = Up_NoShortcut(C_base*2, C_base, bilinear)
        self.up4 = Up_NoShortcut(C_base, C_base, bilinear)
        self.outc = nn.Conv2d(C_base, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.outc(x)
        return x
    
net_no_shortcut = UNet_NoShortcut(n_channels=1, n_classes=3, C_base=32)
optimizer = torch.optim.Adam(net_no_shortcut.parameters(), lr=0.01)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)
solver_no_shortcut = lab.Solver(
    model=net_no_shortcut,
    optimizer=torch.optim.Adam(net_no_shortcut.parameters(), lr=0.01),
    criterion=MyBinaryCrossEntropy(),
    lr_scheduler=lr_scheduler
)

solver_no_shortcut.train(
    epochs=50,
    data_loader=dataloader_train,
    val_loader=dataloader_val,
    save_path='./model_no_shortcut_unet.pth',
    img_name='no_shortcut_unet'
)

dice_scores_no_shortcut = []

net_no_shortcut.to(device)

for images, labels in dataloader_test:
    images = images.to(device)  
    labels = labels.to(device) 
    preds = net_no_shortcut(images)
    preds = torch.sigmoid(preds) > 0.5  # Applying threshold to get binary output
    labels = convert_to_multi_labels(labels)  # Convert labels to multi-label format

    dice_rv_no_shortcut = get_DC(preds[:, 0, :, :], labels[:, 0, :, :])
    dice_myo_no_shortcut = get_DC(preds[:, 1, :, :], labels[:, 1, :, :])
    dice_lv_no_shortcut = get_DC(preds[:, 2, :, :], labels[:, 2, :, :])

    dice_scores_no_shortcut.append((dice_rv_no_shortcut, dice_myo_no_shortcut, dice_lv_no_shortcut))

mean_dice_rv_no_shortcut = np.mean([score[0] for score in dice_scores_no_shortcut])
std_dice_rv_no_shortcut = np.std([score[0] for score in dice_scores_no_shortcut])
mean_dice_myo_no_shortcut = np.mean([score[1] for score in dice_scores_no_shortcut])
std_dice_myo_no_shortcut = np.std([score[1] for score in dice_scores_no_shortcut])
mean_dice_lv_no_shortcut = np.mean([score[2] for score in dice_scores_no_shortcut])
std_dice_lv_no_shortcut = np.std([score[2] for score in dice_scores_no_shortcut])

print(f'RV Dice Coefficient Without Shortcut: Mean={mean_dice_rv_no_shortcut}, SD={std_dice_rv_no_shortcut}')
print(f'MYO Dice Coefficient Without Shortcut: Mean={mean_dice_myo_no_shortcut}, SD={std_dice_myo_no_shortcut}')
print(f'LV Dice Coefficient Without Shortcut: Mean={mean_dice_lv_no_shortcut}, SD={std_dice_lv_no_shortcut}')


transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10)
    ])

class SegmentationDataset(data.Dataset):
    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        image = self.inputs[idx]
        label = self.labels[idx]
        
        if self.transform: 
            all = torch.stack((image, label), dim = 0)
            all = self.transform(all)
            image = all[0]
            label = all[1]
            
        return image, label


def extract_inputs_labels(dataset):
    inputs = []
    labels = []
    for data in dataset:
        input, label = data
        inputs.append(input)
        labels.append(label)
    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    return inputs, labels

inputs_train, labels_train = extract_inputs_labels(train_set)
inputs_val, labels_val = extract_inputs_labels(val_set)
inputs_test, labels_test = extract_inputs_labels(test_set)

train_dataset = SegmentationDataset(inputs_train, labels_train, transform=transform)
val_dataset = SegmentationDataset(inputs_val, labels_val, transform=None)


dataloader_train_aug = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
dataloader_val_aug = data.DataLoader(val_dataset, batch_size=32, shuffle=False)

net_data_aug = UNet(n_channels=1, n_classes=3, C_base=32)
optimizer = torch.optim.Adam(net_data_aug.parameters(), lr=0.01)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

solver = lab.Solver(
    model=net_data_aug,
    optimizer=optimizer,
    criterion=MyBinaryCrossEntropy(),
    lr_scheduler=lr_scheduler,
)

solver.train(
    epochs=50,
    data_loader=dataloader_train_aug,
    val_loader=dataloader_val_aug,
    save_path='./model_original_unet_data_aug_acc.pth',
    img_name='origianl_unet_data_aug_dice_acc'
)

accuracy_scores = []

net_data_aug.to(device)

for images, labels in dataloader_test:
    images = images.to(device)  
    labels = labels.to(device) 
    preds = net_data_aug(images)
    preds = torch.sigmoid(preds) > 0.5  # Applying threshold to get binary output
    labels = convert_to_multi_labels(labels)  # Convert labels to multi-label format

    accuracy_rv = get_accuracy(preds[:, 0, :, :], labels[:, 0, :, :])
    accuracy_myo = get_accuracy(preds[:, 1, :, :], labels[:, 1, :, :])
    accuracy_lv = get_accuracy(preds[:, 2, :, :], labels[:, 2, :, :])

    accuracy_scores.append((accuracy_rv, accuracy_myo, accuracy_lv))

mean_accuracy_rv = np.mean([score[0] for score in accuracy_scores])
std_accuracy_rv = np.std([score[0] for score in accuracy_scores])
mean_accuracy_myo = np.mean([score[1] for score in accuracy_scores])
std_accuracy_myo = np.std([score[1] for score in accuracy_scores])
mean_accuracy_lv = np.mean([score[2] for score in accuracy_scores])
std_accuracy_lv = np.std([score[2] for score in accuracy_scores])

print(f'RV Accuracy: Mean={mean_accuracy_rv}, SD={std_accuracy_rv}')
print(f'MYO Accuracy: Mean={mean_accuracy_myo}, SD={std_accuracy_myo}')
print(f'LV Accuracy: Mean={mean_accuracy_lv}, SD={std_accuracy_lv}')


dice_scores_data_aug = []

net_data_aug.to(device)

for images, labels in dataloader_test:
    images = images.to(device)  
    labels = labels.to(device) 
    preds = net_data_aug(images)
    preds = torch.sigmoid(preds) > 0.5  # Applying threshold to get binary output
    labels = convert_to_multi_labels(labels)  # Convert labels to multi-label format

    dice_rv_data_aug = get_DC(preds[:, 0, :, :], labels[:, 0, :, :])
    dice_myo_data_aug = get_DC(preds[:, 1, :, :], labels[:, 1, :, :])
    dice_lv_data_aug = get_DC(preds[:, 2, :, :], labels[:, 2, :, :])

    dice_scores_data_aug.append((dice_rv_data_aug, dice_myo_data_aug, dice_lv_data_aug))

mean_dice_rv_data_aug = np.mean([score[0] for score in dice_scores_data_aug])
std_dice_rv_data_aug= np.std([score[0] for score in dice_scores_data_aug])
mean_dice_myo_data_aug = np.mean([score[1] for score in dice_scores_data_aug])
std_dice_myo_data_aug = np.std([score[1] for score in dice_scores_data_aug])
mean_dice_lv_data_aug = np.mean([score[2] for score in dice_scores_data_aug])
std_dice_lv_data_aug = np.std([score[2] for score in dice_scores_data_aug])

print(f'RV Dice Coefficient With Data Augmentation: Mean={mean_dice_rv_data_aug}, SD={std_dice_rv_data_aug}')
print(f'MYO Dice Coefficient With Data Augmentation: Mean={mean_dice_myo_data_aug}, SD={std_dice_myo_data_aug}')
print(f'LV Dice Coefficient With Data Augmentation: Mean={mean_dice_lv_data_aug}, SD={std_dice_lv_data_aug}')



class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # input is logits
        targets = convert_to_multi_labels(targets)  # Convert labels to multi-label format

        intersection = (inputs * targets).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        dice_coefficient = (2. * intersection + self.smooth) / (union + self.smooth)

        dice_loss = 1 - dice_coefficient.mean()

        return dice_loss

net_soft_dice = UNet(n_channels=1, n_classes=3, C_base=32)
optimizer = torch.optim.Adam(net_soft_dice.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)


dice_loss = SoftDiceLoss()

solver = lab.Solver(
    model=net_soft_dice,
    optimizer=optimizer,
    criterion=dice_loss,
    lr_scheduler=lr_scheduler,
)


solver.train(
    epochs=50,
    data_loader=dataloader_train_aug,
    val_loader=dataloader_val_aug,
    save_path='./model_soft_dice_loss.pth',
    img_name='soft_dice_loss'
)

accuracy_scores_soft_dice = []

net_soft_dice.to(device)

for images, labels in dataloader_test:
    images = images.to(device)  
    labels = labels.to(device) 
    preds = net_soft_dice(images)
    preds = torch.sigmoid(preds) > 0.5  # Applying threshold to get binary output
    labels = convert_to_multi_labels(labels)  # Convert labels to multi-label format

    accuracy_rv_soft_dice = get_accuracy(preds[:, 0, :, :], labels[:, 0, :, :])
    accuracy_myo_soft_dice = get_accuracy(preds[:, 1, :, :], labels[:, 1, :, :])
    accuracy_lv_soft_dice = get_accuracy(preds[:, 2, :, :], labels[:, 2, :, :])

    accuracy_scores_soft_dice.append((accuracy_rv_soft_dice, accuracy_myo_soft_dice, accuracy_lv_soft_dice))

mean_accuracy_rv_soft_dice = np.mean([score[0] for score in accuracy_scores_soft_dice])
std_accuracy_rv_soft_dice = np.std([score[0] for score in accuracy_scores_soft_dice])
mean_accuracy_myo_soft_dice = np.mean([score[1] for score in accuracy_scores_soft_dice])
std_accuracy_myo_soft_dice = np.std([score[1] for score in accuracy_scores_soft_dice])
mean_accuracy_lv_soft_dice  = np.mean([score[2] for score in accuracy_scores_soft_dice])
std_accuracy_lv_soft_dice  = np.std([score[2] for score in accuracy_scores_soft_dice])

print(f'RV Accuracy With Soft Dice Loss: Mean={mean_accuracy_rv_soft_dice}, SD={std_accuracy_rv_soft_dice}')
print(f'MYO Accuracy With Soft Dice Loss: Mean={mean_accuracy_myo_soft_dice}, SD={std_accuracy_myo_soft_dice}')
print(f'LV Accuracy With Soft Dice Loss: Mean={mean_accuracy_lv_soft_dice}, SD={std_accuracy_lv_soft_dice}')

with open('output_results.txt', 'w') as file:
    file.write(f'RV Dice Coefficient: Mean={mean_dice_rv}, SD={std_dice_rv}\n')
    file.write(f'MYO Dice Coefficient: Mean={mean_dice_myo}, SD={std_dice_myo}\n')
    file.write(f'LV Dice Coefficient: Mean={mean_dice_lv}, SD={std_dice_lv}\n')

    file.write(f'RV Dice Coefficient Without Shortcut: Mean={mean_dice_rv_no_shortcut}, SD={std_dice_rv_no_shortcut}\n')
    file.write(f'MYO Dice Coefficient Without Shortcut: Mean={mean_dice_myo_no_shortcut}, SD={std_dice_myo_no_shortcut}\n')
    file.write(f'LV Dice Coefficient Without Shortcut: Mean={mean_dice_lv_no_shortcut}, SD={std_dice_lv_no_shortcut}\n')

    file.write(f'RV Dice Coefficient With Data Augmentation: Mean={mean_dice_rv_data_aug}, SD={std_dice_rv_data_aug}\n')
    file.write(f'MYO Dice Coefficient With Data Augmentation: Mean={mean_dice_myo_data_aug}, SD={std_dice_myo_data_aug}\n')
    file.write(f'LV Dice Coefficient With Data Augmentation: Mean={mean_dice_lv_data_aug}, SD={std_dice_lv_data_aug}\n')
    
    file.write(f'RV Accuracy: Mean={mean_accuracy_rv}, SD={std_accuracy_rv}\n')
    file.write(f'MYO Accuracy: Mean={mean_accuracy_myo}, SD={std_accuracy_myo}\n')
    file.write(f'LV Accuracy: Mean={mean_accuracy_lv}, SD={std_accuracy_lv}\n')
    
    file.write(f'RV Accuracy With Soft Dice Loss: Mean={mean_accuracy_rv_soft_dice}, SD={std_accuracy_rv_soft_dice}\n')
    file.write(f'MYO Accuracy With Soft Dice Loss: Mean={mean_accuracy_myo_soft_dice}, SD={std_accuracy_myo_soft_dice}\n')
    file.write(f'LV Accuracy With Soft Dice Loss: Mean={mean_accuracy_lv_soft_dice}, SD={std_accuracy_lv_soft_dice}\n')