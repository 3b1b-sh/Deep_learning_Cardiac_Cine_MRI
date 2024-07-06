import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.nn import MSELoss
from torch.utils import data
from torchvision import transforms

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def convert_to_multi_labels(label):
    device = label.device
    B, C, H, W = label.shape
    new_tensor = torch.zeros((B, 3, H, W), device=device)
    mask1 = (label >= 255).squeeze(1)
    mask2 = ((label >= 170) & (label < 255-1)).squeeze(1)
    mask3 = ((label >= 85) & (label < 170-1)).squeeze(1)
    one = torch.ones(size= (B, H, W), device=device)
    zero = torch.zeros(size=(B, H, W), device=device)
    # print(mask1.shape, one.shape, new_tensor.shape)
    new_tensor[:, 0, :, :] = torch.where(mask1, one, zero)
    new_tensor[:, 1, :, :] = torch.where(mask2, one, zero)
    new_tensor[:, 2, :, :] = torch.where(mask3, one, zero)
    return new_tensor


def process_data():
    path = "./cine_seg.npz"
    dataset = np.load(path, allow_pickle=True)
    files = dataset.files
    inputs = []
    labels = []
    for file in files:
        inputs.append(dataset[file][0])
        labels.append(dataset[file][1])
    inputs = torch.Tensor(inputs)
    labels = torch.Tensor(labels)
    return inputs, labels


from torch.utils.data import TensorDataset, DataLoader
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
            all = torch.stack((image, label), dim=0)
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




inputs, labels = process_data()
inputs = inputs.unsqueeze(1)
labels = labels.unsqueeze(1)
dataset = TensorDataset(inputs, labels)
train_size = int(4/7 * len(dataset))
val_size = int(1/7 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

inputs_train, labels_train = extract_inputs_labels(train_set)
inputs_val, labels_val = extract_inputs_labels(val_set)
inputs_test, labels_test = extract_inputs_labels(test_set)

transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10)
    ])


train_dataset = SegmentationDataset(inputs_train, labels_train, transform=transform)
val_dataset = SegmentationDataset(inputs_val, labels_val, transform=None)

dataloader_train_aug = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
dataloader_val_aug = data.DataLoader(val_dataset, batch_size=32, shuffle=False)

print(inputs.shape)
print(labels.shape)
# dataset = TensorDataset(inputs, labels)
#
batch_size = 32
# train_size = int(4 / 7 * len(dataset))
# val_size = int(1 / 7 * len(dataset))
# test_size = len(dataset) - train_size - val_size
# train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
# print(len(train_set), len(val_set), len(test_set))
dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(val_set, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(test_set, batch_size=batch_size, shuffle=True)

from torch import nn
import torch.nn.functional as F


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


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, skip_connection):
        g1 = self.W_g(g)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return skip_connection * psi

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
        self.attention = AttentionBlock(F_g=in_channels // 2, F_l=in_channels // 2, F_int=in_channels // 4)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        #
        # x = torch.cat([x2, x1], dim=1)
        x3 = self.attention(x1, x2)
        x = torch.cat([x3, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, C_base=64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.C_base = C_base

        self.inc = DoubleConv(n_channels, C_base)
        self.down1 = Down(C_base, C_base * 2)
        self.down2 = Down(C_base * 2, C_base * 4)
        self.down3 = Down(C_base * 4, C_base * 8)
        self.down4 = Down(C_base * 8, C_base * 8)
        self.up1 = Up(C_base * 16, C_base * 4, bilinear)
        self.up2 = Up(C_base * 8, C_base * 2, bilinear)
        self.up3 = Up(C_base * 4, C_base, bilinear)
        self.up4 = Up(C_base * 2, C_base, bilinear)
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
        seg_gt_probs = convert_to_multi_labels(seg_gt)
        loss = self.bce(pred_seg_probs, seg_gt_probs)
        return loss



from bme1312.evaluation import get_DC



# def dice_coef_loss(y_pred, y_true):
#     # print(y_true.shape)
#     y_true_multi = convert_to_multi_labels(y_true)
#     # print(y_true_multi.shape)
#     loss = torch.zeros(1, device=y_pred.device)
#     for i in range(3):
#         loss += get_DC(y_pred[:, i, :, :], y_true_multi[:, i, :, :], threshold=0.5)
#     return (1. - loss/3).requires_grad_()

# def dice_coef_loss(y_pred, y_true):
#     # print(y_true.shape)
#     y_true_multi = convert_to_multi_labels(y_true)
#     y_true_multi = torch.argmax(y_true_multi, dim=1, keepdim=True)
#     # print(y_true_multi.shape)
#     # loss = torch.zeros(1, device=y_pred.device)
#     # for i in range(3):
#         # loss += get_DC(y_pred[:, i, :, :], y_true_multi[:, i, :, :], threshold=0.5)
#     y_pred_probs = torch.argmax(y_pred, dim=1, keepdim=True)
#     return (1. - torch.tensor(get_DC(y_pred_probs, y_true_multi), requires_grad=True))

def dice_coef_loss(y_pred, y_true):
    # print(y_true.shape)
    # y_true_multi = convert_to_multi_labels(y_true)
    # print(y_true_multi.shape)
    # loss = torch.zeros(1, device=y_pred.device)
    # for i in range(3):
        # loss += get_DC(y_pred[:, i, :, :], y_true_multi[:, i, :, :])
    y_pred_probs = torch.max(y_pred, dim=1).values
    return (1. - torch.tensor(get_DC(y_pred_probs, y_true), requires_grad=True))

def generalized_dice_coeff(y_pred, y_true):
    Ncl = y_pred.shape[1]
    y_true_prob = convert_to_multi_labels(y_true)
    w = torch.zeros(size=(Ncl,))
    w = torch.sum(y_true_prob, dim=(0,2,3))
    w = 1/(w**2+0.000001)
    # Compute gen dice coef:
    numerator = y_true_prob*y_pred
    numerator = w*torch.sum(numerator, dim=(0,1,2,3))
    numerator = torch.sum(numerator)
    denominator = y_true_prob+y_pred
    denominator = w*torch.sum(denominator, dim=(0,1,2,3))
    denominator = torch.sum(denominator)
    gen_dice_coef = 2*numerator/denominator
    return gen_dice_coef


def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice_coeff(y_true, y_pred)

import bme1312.lab2 as lab

net = UNet(n_channels=1, n_classes=3, C_base=32)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

solver = lab.Solver(
    model=net,
    optimizer=optimizer,
    criterion=MyBinaryCrossEntropy(),
    lr_scheduler=lr_scheduler
)
solver.train(
    epochs=30,
    data_loader=dataloader_train_aug,
    val_loader=dataloader_val_aug,
    save_path='./model_attention_30.pth',
    img_name='attention_aug_30',
)

solver.visualize(data_loader=dataloader_test, idx=10)
