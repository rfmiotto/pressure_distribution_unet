from typing import Optional
import torch


def save_checkpoint(
    model, optimizer, epoch, loss, best_acc, filename="my_checkpoint.pth.tar"
):
    print("=> Saving checkpoint")

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "best_acc": best_acc,
    }

    torch.save(state, filename)


def load_checkpoint(
    device,
    model,
    optimizer: Optional[torch.optim.Optimizer] = None,
    filename="my_checkpoint.pth.tar",
):
    print("=> Loading checkpoint")

    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    best_acc = checkpoint["best_acc"]

    return epoch, loss, best_acc
