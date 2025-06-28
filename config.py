# config.py

import os
import torch

class Config:
    BASE_PATH        = 'Balanced/'

    IMAGE_HEIGHT     = 128
    IMAGE_WIDTH      = 128
    NUM_CLASSES      = 6

    BATCH_SIZE       = 16
    NUM_EPOCHS       = 100
    LEARNING_RATE    = 1e-4
    WEIGHT_DECAY     = 1e-5

    TRAIN_IMG_DIR    = os.path.join(BASE_PATH, 'train/images')
    TRAIN_MASK_DIR   = os.path.join(BASE_PATH, 'train/masks')
    VAL_IMG_DIR      = os.path.join(BASE_PATH, 'val/images')
    VAL_MASK_DIR     = os.path.join(BASE_PATH, 'val/masks')

    DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_SAVE_PATH  = 'best_weed_model.pth'

    NUM_WORKERS      = 2
    PIN_MEMORY       = True