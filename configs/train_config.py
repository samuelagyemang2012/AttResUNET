import torch

# Hyperparameters etc.
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
ADAM_BETAS = (0.9, 0.99)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
INTERVAL = 10
NUM_EPOCHS = 500
NUM_WORKERS = 0
IMAGE_HEIGHT = 400  # 1280 originally
IMAGE_WIDTH = 400  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
RES_DIR = "../res/"
TRAIN_DEG_DIR = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/train/deg/"  # "C:/Users/Administrator/Desktop/datasets/dehaze/reside/OTS/training_data/train/hazy/"
TRAIN_CLEAR_DIR = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/train/clear/"  # "C:/Users/Administrator/Desktop/datasets/dehaze/reside/OTS/training_data/train/clear/"

VAL_DEG_DIR = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/val/deg/"  # "C:/Users/Administrator/Desktop/datasets/dehaze/reside/OTS/training_data/val/hazy/"
VAL_CLEAR_DIR = "C:/Users/Administrator/Desktop/datasets/snow100k/training_data_large/val/clear/"  # "C:/Users/Administrator/Desktop/datasets/dehaze/reside/OTS/training_data/val/clear/"
TOLERANCE = 30

# colab
# RES_DIR = "/content/AttResUNET/res"
# TRAIN_HAZY_DIR = "/content/drive/MyDrive/ML_Data/datasets/images/ITS/train/hazy/"
# TRAIN_CLEAR_DIR = "/content/drive/MyDrive/ML_Data/datasets/images/ITS/train/clear/"
# VAL_HAZY_DIR = "/content/drive/MyDrive/ML_Data/datasets/images/ITS/val/hazy/"
# VAL_CLEAR_DIR = "/content/drive/MyDrive/ML_Data/datasets/images/ITS/val/clear/"
# TOLERANCE = 30"
