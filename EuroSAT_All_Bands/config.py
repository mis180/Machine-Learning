import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import random
MODEL_LIST = ["VGG11", "VGG19", "RESNET18", "RESNET152", "EFFICIENTNET-B1", "EFFICIENTNET-B7"]
MODEL_NAME = "EFFICIENTNET-B7"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
BATCH_SIZE = 256
NUM_EPOCHS = 300
NUM_WORKERS = 2
WEIGHT_DECAY = 1e-5
SEED = 42
SAVE_MODEL = True
LOAD_MODEL = True

CHECKPOINT_FILE =  f'models/EuroSATallBands_{MODEL_NAME}_O.pth.tar'  # Save model
RESULTS_FILE = f"output/csv/EuroSATallBands_{MODEL_NAME}_O.csv"   # Save outputs in CSV
PLOT_FILE =    f"output/png/EuroSATallBands_{MODEL_NAME}_O.png"
# Data Preparation
train_transforms = A.Compose(
    [
        A.Resize(width=64, height=64),

        # Geometry & Topology

        # A.HorizontalFlip(p=0.25),
        # A.VerticalFlip(p=0.25),
        # A.Rotate(limit=random.randint(1, 89), p=0.25),
        # A.Transpose(p=0.25),
        # A.ShiftScaleRotate(p=0.25),
        # A.GridDistortion(p=0.25),

        # Color

        # A.RandomBrightnessContrast(p=0.25),
        # A.RandomGamma(p=0.25),
        # A.Blur(p=0.25),
        # A.Sharpen(p=0.25),

        A.ToFloat(max_value=28003.0,),
        ToTensorV2(),
    ]
)

val_test_transforms = A.Compose(
    [
        A.Resize(width=64, height=64),
        A.ToFloat(max_value=28003),
        ToTensorV2(),
    ]
)