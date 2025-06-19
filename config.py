# config.py
import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    TRAIN_IMG_DIR = "Balanced/train/images"
    TRAIN_MASK_DIR = "Balanced/train/masks"
    VAL_IMG_DIR = "Balanced/val/images"
    VAL_MASK_DIR = "Balanced/val/masks"
    
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8
    NUM_EPOCHS = 200
    NUM_WORKERS = 2
    
    IMAGE_HEIGHT = 224
    IMAGE_WIDTH = 224
    INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    
    PIN_MEMORY = True
    LOAD_MODEL = False
    MODEL_SAVE_PATH = "/content/drive/MyDrive/colab/best_model.pth.tar"