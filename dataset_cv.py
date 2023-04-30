import os
from PIL import Image
from os.path import splitext
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torchvision
import torch
import matplotlib.pyplot as plt
import cv2
from configs import train_config as cfg


class Test(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.deg_images = [image_dir + f for f in os.listdir(image_dir) if
                           f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]

        self.clean_images = [mask_dir + f for f in os.listdir(mask_dir) if
                             f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]

        # Filter the images to ensure they are counterparts of the same scene
        self.filter_files()
        self.size = len(self.deg_images)

        self.transform = transform

    def filter_files(self):
        assert len(self.deg_images) == len(self.clean_images)
        deg_ims = []
        clean_ims = []

        for deg_img_path, clean_img_path in zip(self.deg_images, self.clean_images):

            deg = cv2.imread(deg_img_path)
            clean = cv2.imread(clean_img_path)

            if deg.size == clean.size:
                deg_ims.append(deg_img_path)
                clean_ims.append(clean_img_path)

        self.deg_images = deg_ims
        self.clean_images = clean_ims

    def __getitem__(self, index):
        deg_img = self.rgb_loader(self.deg_images[index])
        clean_img = self.rgb_loader(self.clean_images[index])

        if self.transform is not None:
            augmentations = self.transform(image=deg_img, mask=clean_img)
            deg_img = augmentations["image"]
            clean_img = augmentations["mask"]

        return deg_img, clean_img

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img).astype(np.float32)

            return img

    def __len__(self):
        return self.size


class Data(Dataset):
    def __init__(self, clear_imgs_dir, deg_imgs_dir, scale=1):
        self.clear_imgs_dir = clear_imgs_dir
        self.deg_imgs_dir = deg_imgs_dir
        self.scale = scale
        assert 0 < scale <= 1, "Scale must be between 0 and 1"

        self.clear_images = [
            file for file in os.listdir(clear_imgs_dir) if
            file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg')
        ]

        self.deg_images = [
            file for file in os.listdir(deg_imgs_dir) if
            file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg')
        ]

    def __len__(self):
        return len(self.clear_images)

    @classmethod
    def preprocess(cls, pil_img, scale, image_size):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, "Scale is too small"
        pil_img = pil_img.resize((image_size, image_size))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):

        deg_file = self.deg_imgs_dir + self.deg_images[i]

        clear_file = self.clear_imgs_dir + self.clear_images[i]

        deg_img = Image.open(deg_file)
        clear_img = Image.open(clear_file)

        # assert (
        #         clear_img.size == hazy_img.size
        # ), f"Image and mask {idx} should be the same size, but are {clear_img.size} and {hazy_img.size}"

        clear_img = self.preprocess(clear_img, self.scale, cfg.IMAGE_WIDTH)
        deg_img = self.preprocess(deg_img, self.scale, cfg.IMAGE_WIDTH)

        return (
            torch.from_numpy(clear_img).type(torch.FloatTensor),
            torch.from_numpy(deg_img).type(torch.FloatTensor),
        )


def test():
    def process(arr):
        arr = arr.squeeze(0).permute(1, 2, 0).numpy()
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

        return arr

    def show_images(hazy, clear):
        cv2.imshow("haze", hazy)
        cv2.imshow("clear", clear)
        cv2.waitKey(-1)

    train_transform = A.Compose([
        A.Resize(height=400, width=400),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        # A.Normalize(
        #     mean=[0.0, 0.0, 0.0],
        #     std=[1.0, 1.0, 1.0],
        # max_pixel_value=255.0,
        # ),
        # ToTensorV2()
    ])

    TRAIN_DEG_DIR = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/OTS/training_data/train/hazy/"
    TRAIN_CLEAR_DIR = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/OTS/training_data/train/clear/"

    VAL_HAZY_DIR = "C:/Users/Administrator/Desktop/datasets/SOTs/data/SOTS/val/hazy/"
    VAL_CLEAR_DIR = "C:/Users/Administrator/Desktop/datasets/SOTs/data/SOTS/val/clear/"

    train_dataset = Data(clear_imgs_dir=TRAIN_CLEAR_DIR, deg_imgs_dir=TRAIN_DEG_DIR)  # , transform=train_transform)
    loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=0)

    examples = next(iter(loader))
    clear_images, deg_images = examples

    for i, ci in enumerate(clear_images):
        clear_image = process(ci)
        print(clear_image)

        deg_image = process(deg_images[i])
        print(deg_image)

        show_images(deg_image, clear_image)


if __name__ == "__main__":
    test()
