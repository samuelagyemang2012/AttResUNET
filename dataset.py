import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class SOTS(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.hazy_images = [image_dir + f for f in os.listdir(image_dir) if
                            f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]

        self.clean_images = [mask_dir + f for f in os.listdir(mask_dir) if
                             f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]

        # Filter the images to ensure they are counterparts of the same scene
        self.filter_files()
        self.size = len(self.hazy_images)

        self.transform = transform

    def filter_files(self):
        assert len(self.hazy_images) == len(self.clean_images)
        hazy_ims = []
        clean_ims = []
        for hazy_img_path, clean_img_path in zip(self.hazy_images, self.clean_images):
            hazy = Image.open(hazy_img_path)
            clean = Image.open(clean_img_path)
            if hazy.size == clean.size:
                hazy_ims.append(hazy_img_path)
                clean_ims.append(clean_img_path)
        self.hazy_images = hazy_ims
        self.clean_images = clean_ims

    def __getitem__(self, index):
        hazy_img = self.rgb_loader(self.hazy_images[index])
        clean_img = self.rgb_loader(self.clean_images[index])

        if self.transform is not None:
            augmentations = self.transform(image=hazy_img, mask=clean_img)
            hazy_img = augmentations["image"]
            clean_img = augmentations["mask"]

        return hazy_img, clean_img

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = np.array(Image.open(f).convert('RGB')).astype(np.float32)
            # img = np.transpose(img, (2, 0, 1))
            return img
            # return img.transpose((2, 0, 1))

    def __len__(self):
        return self.size

    # def __getitem__(self, index):
    #     hazy_img_path = os.path.join(self.hazy_img_dir, self.images[index])
    #     clear_img_path = os.path.join(self.clear_img_dir, self.images[index])
    #     hazy_image = np.array(Image.open(hazy_img_path).convert("RGB"))
    #     clear_image = np.array(Image.open(clear_img_path).convert("RGB"))
    #
    #     if self.transform is not None:
    #         augmentations = self.transform(image=hazy_image, mask=clear_image)
    #         hazy_image = augmentations["hazy_image"]
    #         clear_image = augmentations["clear_image"]
    #
    #     return hazy_image, clear_image
