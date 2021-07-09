import config
import sys
import torch
from os import path
import torch.nn as nn
from torch import optim
from utils.dataset import GetDataset
import torchvision.models as models
from torch.utils.data import DataLoader
from utils.utils import (
    load_checkpoint,
)
import matplotlib.pyplot as plt
from PIL import Image


def plot_imeges(paths, pred, y):
    LABELS = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop',
              'Residential', 'River', 'SeaLake']
    fig = plt.figure(figsize=(15, 10))
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        ax.imshow(Image.open(paths[i], 'r'))
        plt.title(paths[i].split('/')[-2])
        plt.tight_layout()
        ax.set_title("Actual: {}\n(Predicted: {})".format(LABELS[y[i]], LABELS[pred[i]]),
                             color=("green" if pred[i] == y[i] else "red"))
    plt.show()
    plt.savefig(f"output/EuroSAT_Resutls.png")

# Load Data
test_dataset = GetDataset(
        images_folder='data/EuroSAT/',
        path_to_csv='../data/test.csv',
        transform=config.val_test_transforms,)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, num_workers=config.NUM_WORKERS, shuffle=True, pin_memory=True)


if __name__ == "__main__":
    # Make model similar to Train
    if config.MODEL_NAME=="VGG16":
        model = models.vgg16(pretrained=True)
        num_input = model.classifier._modules['6'].in_features # We are updating last layer
        model.classifier._modules['6'] = nn.Linear(num_input, 10)
        model.to(config.DEVICE)
    elif config.MODEL_NAME == "RESNET18":
        model = models.resnet18(pretrained=True)
        num_input = model.fc.in_features
        model.fc = nn.Linear(num_input, 10)
        model.to(config.DEVICE)
    else:
        print("Model was not selected")
        sys.exit()

    filename = config.CHECKPOINT_FILE

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, momentum=config.MOMENTUM,)

    if config.LOAD_MODEL and path.exists(config.CHECKPOINT_FILE):
        print("Checkpoint is loaded:")
        load_checkpoint(torch.load(config.CHECKPOINT_FILE), model, optimizer, config.LEARNING_RATE)
    model.eval()
    with torch.no_grad():
        for x, y, path in test_loader:
            x = x.to(device=config.DEVICE)
            y = y.to(device=config.DEVICE)
            scores = model(x)
            _, pred = scores.max(1)
            plot_imeges(path, pred, y)
            break