import torch

# Hyperparameters etc.
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
INTERVAL = 10
NUM_EPOCHS = 300
NUM_WORKERS = 0
IMAGE_HEIGHT = 400  # 1280 originally
IMAGE_WIDTH = 400  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
RES_DIR = "res/"
TRAIN_HAZY_DIR = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/ITS/training_data_small/train/hazy/"  # "C:/Users/Administrator/Desktop/datasets/dehaze/reside/SOTs/training_data/SOTS/train/hazy/"
TRAIN_CLEAR_DIR = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/ITS/training_data_small/train/clear/"  # "C:/Users/Administrator/Desktop/datasets/dehaze/reside/SOTs/training_data/SOTS/train/clear/"
VAL_HAZY_DIR = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/ITS/training_data_small/val/hazy/"  # "C:/Users/Administrator/Desktop/datasets/dehaze/reside/SOTs/training_data/SOTS/val/hazy/"
VAL_CLEAR_DIR = "C:/Users/Administrator/Desktop/datasets/dehaze/reside/ITS/training_data_small/val/clear/"  # C:/Users/Administrator/Desktop/datasets/dehaze/reside/SOTs/training_data/SOTS/val/clear/"
TOLERANCE = 30
