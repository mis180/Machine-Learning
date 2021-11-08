import torch
import config
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_one_epoch(epoch, loader, model, optimizer, loss_fn , device):
    model.train()
    losses = []
    loop = tqdm(enumerate(loader), total=len(loader), leave=True)
    for batch_idx, (data, targets, _) in loop:
        data = data.to(device=device)
        targets = targets.to(device=device)
        # forward
        pred = model(data)
        loss = loss_fn(pred, targets)
        # backward
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss)
        optimizer.step()
        # update progres bar
        loop.set_description(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
        loop.set_postfix(loss=loss.item())
    return sum(losses)/len(losses)


def load_checkpoint(checkpoint, model, optimizer, lr):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def check_accuracy(loader, model,  text=None, eval=True):
    num_correct = 0
    num_samples = 0
    if eval:
        model.eval()  # set models to evaluation mode
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device=config.DEVICE)
            y = y.to(device=config.DEVICE, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = (float(num_correct) / num_samples) * 100.0
        print(" %s:  Got %d / %d correct (%.2f)" % (text, num_correct, num_samples, acc))
        return acc


def create_plot(plot_stuff):
    df = pd.DataFrame.from_dict(plot_stuff, orient="index")
    df.to_csv(config.RESULTS_FILE)
    plt.plot(plot_stuff['train_acc_list'], 'r-')
    plt.plot(plot_stuff['val_acc_list'], 'b-')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Validation'])
    plt.title('Train Accuracy vs Test Accuracy')
    plt.grid(True)
    plt.savefig(config.PLOT_FILE)
    # plt.show()
#
