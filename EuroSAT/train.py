import sys
import config
import torch
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)
from os import path
import torch.nn as nn
from torch import optim
from utils.dataset import GetDataset
import torchvision.models as models
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from utils.utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    train_one_epoch,
    create_plot,)

plot_stuff = {'device': config.DEVICE,'epochs': config.NUM_EPOCHS,
              'lr': config.LEARNING_RATE,'momentum': config.MOMENTUM,
              'weight_decay':config.WEIGHT_DECAY, 'model_name':config.MODEL_NAME,
              "loss_list": [], 'train_acc_list': [], 'val_acc_list': [],'test_acc': 0,}

# Load Data
train_dataset = GetDataset(
        images_folder='data/EuroSAT/',
        path_to_csv='data/train.csv',
        transform=config.train_transforms,)

val_dataset = GetDataset(
        images_folder='data/EuroSAT/',
        path_to_csv='data/validation.csv',
        transform=config.val_test_transforms,)

test_dataset = GetDataset(
        images_folder='data/EuroSAT/',
        path_to_csv='data/test.csv',
        transform=config.val_test_transforms,)

train_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True, pin_memory=True)

if __name__ == "__main__":
    if config.MODEL_NAME=="VGG11":
        print("Model name: VGG11")
        model = models.vgg11(pretrained=True)
        num_input = model.classifier._modules['6'].in_features # We are updating last layer
        model.classifier._modules['6'] = nn.Linear(num_input, 10)
        model.to(config.DEVICE)
        # print(model)
    elif config.MODEL_NAME=="VGG19":
        print("Model name: VGG19")
        model = models.vgg11(pretrained=True)
        num_input = model.classifier._modules['6'].in_features # We are updating last layer
        model.classifier._modules['6'] = nn.Linear(num_input, 10)
        model.to(config.DEVICE)
        # print(model)
    elif config.MODEL_NAME == "RESNET18":
        print("Model name: RESNET18")
        model = models.resnet18(pretrained=True)
        num_input = model.fc.in_features
        model.fc = nn.Linear(num_input, 10)
        model.to(config.DEVICE)
        # print(model)
    elif config.MODEL_NAME == "RESNET152":
        print("Model name: RESNET152")
        model = models.resnet152(pretrained=True)
        num_input = model.fc.in_features
        model.fc = nn.Linear(num_input, 10)
        model.to(config.DEVICE)
        # print(model)
    elif config.MODEL_NAME == "EFFICIENTNET-B1":
        print("Model name: EFFICIENTNET-B1")
        model = EfficientNet.from_pretrained("efficientnet-b1",  num_classes=10)
        model.to(config.DEVICE)
        # print(model)
    elif config.MODEL_NAME == "EFFICIENTNET-B7":
        print("Model name: EFFICIENTNET-B7")
        model = EfficientNet.from_pretrained("efficientnet-b7",  num_classes=10)
        model.to(config.DEVICE)
        # print(model)
    else:
        print("Model was not selected")
        sys.exit()
    filename = config.CHECKPOINT_FILE

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=5, verbose=True)
    if config.LOAD_MODEL and path.exists(config.CHECKPOINT_FILE):
        print("Checkpoint is loaded:")
        load_checkpoint(torch.load(config.CHECKPOINT_FILE), model, optimizer, config.LEARNING_RATE)

    for epoch in range(config.NUM_EPOCHS):
        mean_loss = train_one_epoch(epoch, train_loader, model, optimizer, loss_fn, config.DEVICE)
        scheduler.step(mean_loss)
        #Cheking the accuracy of Train and Val for the plot
        train_acc = check_accuracy(train_loader, model, text="Train Dataset", eval=True)
        val_acc = check_accuracy(val_loader, model, text="Val Dataset", eval=True)
        # Storing the The Accuracy and Loss
        plot_stuff['loss_list'].append(mean_loss.item())
        plot_stuff['train_acc_list'].append(train_acc)
        plot_stuff['val_acc_list'].append(val_acc)

        if config.SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
    save_checkpoint(checkpoint, filename=filename)
    test_acc = check_accuracy(test_loader, model, text='Test Dataset', eval=False)
    plot_stuff['test_acc'] = test_acc
    print("Accuracy of Test dataset: ", test_acc)
    create_plot(plot_stuff)
