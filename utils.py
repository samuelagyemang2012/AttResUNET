import torch
import torchvision
from torch.utils.data import DataLoader


def save_checkpoint(state, filename="dehaze_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

    return model


def get_loaders(dataset, train_dir, train_dir2, val_dir, val_dir2, batch_size, train_transform, val_transform,
                num_workers=4, pin_memory=True):
    train_ds = dataset(
        image_dir=train_dir,
        mask_dir=train_dir2,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = dataset(
        image_dir=val_dir,
        mask_dir=val_dir2,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader




