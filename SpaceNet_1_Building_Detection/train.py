import torch
import config
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

TRAIN_IMG_DIR = "data/rio_train/"
TRAIN_MASK_DIR = "data/rio_mask/"
VAL_IMG_DIR = "data/rio_val/"
VAL_MASK_DIR = "data/rio_mask/"

# print("Device:", config.DEVICE)

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=config.DEVICE)
        targets = targets.float().unsqueeze(1).to(device=config.DEVICE)
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    model = UNET(in_channels=3, out_channels=1).to(config.DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=5, verbose=True)
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,

        config.BATCH_SIZE,
        config.train_transform,
        config.val_transforms,
        config.NUM_WORKERS,
        config.PIN_MEMORY,
    )
    if config.LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=config.DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=config.DEVICE)


    # print some examples to a folder
    save_predictions_as_imgs(
    val_loader, model, folder="saved_images/", device=config.DEVICE )

if __name__ == "__main__":
    main()