import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


class SOTS(Dataset):
    def __init__(self, hazy_image_dir, clear_image_dir, transform=None):
        self.hazy_img_dir = hazy_image_dir
        self.clear_img_dir = clear_image_dir
        self.transform = transform
        self.images = os.listdir(hazy_image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        hazy_img_path = os.path.join(self.hazy_img_dir, self.images[index])
        clear_img_path = os.path.join(self.clear_img_dir, self.images[index])
        hazy_image = np.array(Image.open(hazy_img_path).convert("RGB"))
        clear_image = np.array(Image.open(clear_img_path).convert("RGB"))

        if self.transform is not None:
            augmentations = self.transform(image=hazy_image, mask=clear_image)
            hazy_image = augmentations["hazy_image"]
            clear_image = augmentations["clear_image"]

        return hazy_image, clear_image
