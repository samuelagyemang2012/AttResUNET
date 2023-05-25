import os
from PIL import Image
from os.path import splitext
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch
import cv2
from configs import train_config as cfg
from torchvision import transforms
import torchvision.transforms.functional as TF
import random


class ImageDataset(Dataset):
    def __init__(self, clear_imgs_dir, deg_imgs_dir, resize_dim=None):
        self.clear_imgs_dir = clear_imgs_dir
        self.deg_imgs_dir = deg_imgs_dir
        self.resize_dim = resize_dim

        self.clear_images = [
            file for file in os.listdir(clear_imgs_dir) if
            file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg')
        ]

        self.deg_images = [
            file for file in os.listdir(deg_imgs_dir) if
            file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg')
        ]

        self.both_transform = transforms.Compose(
            [
                A.RandomCrop(width=cfg.PATCH_SIZE, height=cfg.PATCH_SIZE),
                transforms.ToTensor(),
            ]
        )

    def transform(self, clear_image, deg_image, crop_size):
        # Resize
        if self.resize_dim is not None:
            resize = transforms.Resize(size=(cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
            clear_image = resize(clear_image)
            deg_image = resize(deg_image)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(clear_image, output_size=(crop_size, crop_size))

        clear_image = TF.crop(clear_image, i, j, h, w)
        deg_image = TF.crop(deg_image, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            clear_image = TF.hflip(clear_image)
            deg_image = TF.hflip(deg_image)

        # Random vertical flipping
        if random.random() > 0.5:
            clear_image = TF.vflip(clear_image)
            deg_image = TF.vflip(deg_image)

        # Transform to tensor
        clear_image = TF.to_tensor(clear_image)
        deg_image = TF.to_tensor(deg_image)
        return clear_image, deg_image

    def __getitem__(self, index):

        clear_img = Image.open(self.clear_imgs_dir + self.clear_images[index]).convert("RGB")
        deg_img = Image.open(self.deg_imgs_dir + self.deg_images[index]).convert("RGB")

        img_clear, img_deg = self.transform(clear_img, deg_img, crop_size=cfg.PATCH_SIZE)

        return img_clear.type(torch.float32), img_deg.type(torch.float32)

    def __len__(self):
        return len(self.clear_images)


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

    TRAIN_DEG_DIR = "C:/Users/Administrator/Desktop/datasets/snow100k/preds/snow_sr/"
    TRAIN_CLEAR_DIR = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/test2/clear/"

    train_dataset = ImageDataset(clear_imgs_dir=TRAIN_CLEAR_DIR, deg_imgs_dir=TRAIN_DEG_DIR, resize_dim=400)
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
