import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    TRAIN_IMG_DIR = "Balanced/train/images"
    TRAIN_MASK_DIR = "Balanced/train/masks"
    VAL_IMG_DIR = "Balanced/val/images"
    VAL_MASK_DIR = "Balanced/val/masks"
    
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    NUM_WORKERS = 2
    
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    
    PIN_MEMORY = True
    LOAD_MODEL = False
    BASE_PATH = "/content/drive/MyDrive/colab/"
    MODEL_SAVE_PATH = BASE_PATH + "best_model.pth.tar"
    PERFORMANCE_PATH = BASE_PATH + "rendimiento_miou.png"

    NUM_SPECIAL_AUGMENTATIONS = 15