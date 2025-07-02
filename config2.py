# config.py

import torch

class Config:
    DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

    TRAIN_IMG_DIR  = "Balanced/train/images"
    TRAIN_MASK_DIR = "Balanced/train/masks"
    VAL_IMG_DIR    = "Balanced/val/images"
    VAL_MASK_DIR   = "Balanced/val/masks"

    LEARNING_RATE  = 1e-2       # 0.01 según PDF
    BATCH_SIZE     = 32         # batch=32
    NUM_EPOCHS     = 60         # 60 épocas
    NUM_WORKERS    = 2
    PIN_MEMORY     = True

    IMAGE_HEIGHT   = 128        # resize a 128×128
    IMAGE_WIDTH    = 128

    BASE_PATH      = "/content/drive/MyDrive/colab/"
    MODEL_SAVE_PATH= BASE_PATH + "best_model.pth.tar"
    PERFORMANCE_PATH = BASE_PATH + "rendimiento_loss.png"
