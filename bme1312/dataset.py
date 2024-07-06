
import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image


class ImageFolder(data.Dataset):
    def __init__(self, root, mode='train', augmentation_prob=0.4):
        assert mode in {'train', 'valid'}
        """Initializes image paths and preprocessing module."""
        self.root = root

        # GT : Ground Truth
        self.GT_paths = root + '_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        filename = image_path.split('_')[-1][:-len(".jpg")]
        GT_path = self.GT_paths + 'case_' + filename + '_segmentation.jpg'

        image = Image.open(image_path)
        seg_gt = Image.open(GT_path)

        aspect_ratio = image.size[1] / image.size[0]

        Transform = []

        ResizeRange = random.randint(300, 320)
        Transform.append(T.Resize((int(ResizeRange * aspect_ratio), ResizeRange)))
        p_transform = random.random()

        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            RotationDegree = random.randint(0, 3)
            RotationDegree = self.RotationDegree[RotationDegree]
            if (RotationDegree == 90) or (RotationDegree == 270):
                aspect_ratio = 1 / aspect_ratio

            Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))

            RotationRange = random.randint(-10, 10)
            Transform.append(T.RandomRotation((RotationRange, RotationRange)))
            CropRange = random.randint(250, 270)
            Transform.append(T.CenterCrop((int(CropRange * aspect_ratio), CropRange)))
            Transform = T.Compose(Transform)

            image = Transform(image)
            seg_gt = Transform(seg_gt)

            ShiftRange_left = random.randint(0, 20)
            ShiftRange_upper = random.randint(0, 20)
            ShiftRange_right = image.size[0] - random.randint(0, 20)
            ShiftRange_lower = image.size[1] - random.randint(0, 20)
            image = image.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
            seg_gt = seg_gt.crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))

            if random.random() < 0.5:
                image = F.hflip(image)
                seg_gt = F.hflip(seg_gt)

            if random.random() < 0.5:
                image = F.vflip(image)
                seg_gt = F.vflip(seg_gt)

            Transform = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02)

            image = Transform(image)

            Transform = []

        Transform.append(T.Resize((int(256 * aspect_ratio) - int(256 * aspect_ratio) % 16, 256)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)

        image = Transform(image)
        seg_gt = Transform(seg_gt)

        Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = Norm_(image)

        Grayscale_ = T.Grayscale(num_output_channels=1)
        image = Grayscale_(image)

        return image, seg_gt

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

class Test_ImageFolder(data.Dataset):
    def __init__(self, root, mode='train', augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        assert mode in {'test'}
        self.root = root

        # GT : Ground Truth
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        filename = image_path.split('_')[-1][:-len(".jpg")]


        image = Image.open(image_path)


        aspect_ratio = image.size[1] / image.size[0]

        Transform = []
        ResizeRange = random.randint(300, 320)
        Transform.append(T.Resize((int(ResizeRange * aspect_ratio), ResizeRange)))

        Transform.append(T.Resize((int(256 * aspect_ratio) - int(256 * aspect_ratio) % 16, 256)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)

        image = Transform(image)

        Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = Norm_(image)

        Grayscale_ = T.Grayscale(num_output_channels=1)
        image = Grayscale_(image)

        return image,filename

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)



def get_loader(image_root_path, batch_size, num_workers=4, mode='train', augmentation_prob=0.4):
    """Builds and returns Dataloader."""
    if mode=='test':
        dataset = Test_ImageFolder(root=image_root_path, mode=mode, augmentation_prob=augmentation_prob)
    else:
        dataset = ImageFolder(root=image_root_path, mode=mode, augmentation_prob=augmentation_prob)

    data_loader = data.DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=True if mode == 'train' else False,
                                num_workers=num_workers)

    return data_loader