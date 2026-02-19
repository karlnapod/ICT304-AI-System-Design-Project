# train.py
# Called using the train.run() from client.py
# or run directly: python train.py

import os
import sys
import json
import queue
import threading
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import config

warnings.filterwarnings("ignore")


# Hyperparameter prompt

def prompt_hyperparams() -> dict:
    """
    Printing each hyperparameter with its current value.
    User presses Enter to keep it, or types a new value.
    Returns a dict of confirmed values.
    """
    defaults = {
        "BATCH_SIZE"  : config.BATCH_SIZE,
        "NUM_EPOCHS"  : config.NUM_EPOCHS,
        "LR"          : config.LR,
        "WEIGHT_DECAY": config.WEIGHT_DECAY,
    }

    descriptions = {
        "BATCH_SIZE"  : "Batch size (e.g. 64 or 32)",
        "NUM_EPOCHS"  : "Number of training epochs",
        "LR"          : "Learning rate (e.g. 2e-4)",
        "WEIGHT_DECAY": "AdamW weight decay (e.g. 1e-2)",
    }

    converters = {
        "BATCH_SIZE"  : int,
        "NUM_EPOCHS"  : int,
        "LR"          : float,
        "WEIGHT_DECAY": float,
    }

    print("\n" + "─" * 60)
    print("  HYPERPARAMETERS")
    print("  Press Enter to keep the current value, or type a new one.")
    print("─" * 60)

    confirmed = {}
    for key, default in defaults.items():
        while True:
            raw = input(f"  {descriptions[key]}\n"
                        f"    {key} [{default}]: ").strip()
            if raw == "":
                confirmed[key] = default
                break
            try:
                confirmed[key] = converters[key](raw)
                break
            except ValueError:
                print(f"Invalid value '{raw}' — please try again.")

    print("\nConfirmed hyperparameters:")
    for k, v in confirmed.items():
        print(f"    {k:<16}: {v}")
    print("─" * 60)
    go = input("\nStart training? [Y/n]: ").strip().lower()
    if go == "n":
        print("  Training cancelled.")
        return None
    return confirmed


# Dataset / DataLoader

class SkinLesionDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])


class ThreadedPrefetcher:
    """Background-thread prefetcher — no multiprocessing, no pickling issues."""

    def __init__(self, dataloader, num_threads: int, queue_size: int, device, pin: bool):
        self.dataloader  = dataloader
        self.num_threads = num_threads
        self.queue_size  = queue_size
        self.device      = device
        self.pin         = pin

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        batch_q     = queue.Queue(maxsize=self.queue_size)
        _DONE       = object()
        loader_iter = iter(self.dataloader)
        iter_lock   = threading.Lock()
        n_done      = [0]
        done_lock   = threading.Lock()

        def worker():
            while True:
                with iter_lock:
                    try:
                        batch = next(loader_iter)
                    except StopIteration:
                        break
                if self.pin and self.device.type == "cuda":
                    batch = tuple(
                        t.pin_memory() if isinstance(t, torch.Tensor) else t
                        for t in batch
                    )
                batch_q.put(batch)
            with done_lock:
                n_done[0] += 1
                if n_done[0] == self.num_threads:
                    batch_q.put(_DONE)

        threads = [threading.Thread(target=worker, daemon=True)
                   for _ in range(self.num_threads)]
        for t in threads:
            t.start()
        while True:
            item = batch_q.get()
            if item is _DONE:
                break
            yield item
        for t in threads:
            t.join()


# Data loading helpers

def load_dataframes():
    """
    Load the pre-saved train/val/test CSVs produced by pre_process.py.
    Fails clearly if they don't exist yet — run preprocessing first.
    """
    train_path = config.BASE_PATH / "train_split.csv"
    val_path   = config.BASE_PATH / "val_split.csv"
    test_path  = config.BASE_PATH / "test_split.csv"

    missing = [p.name for p in [train_path, val_path, test_path] if not p.exists()]
    if missing:
        print("\n The following split CSVs are missing:")
        for name in missing:
            print(f"      {name}")
        print("\n  Run preprocessing first (option 1 in the main menu)")
        print("\n pre_process.py will resize the images and save the split CSVs")
        return None, None, None

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    # Convert filepath strings back to Path objects
    train_df["filepath"] = train_df["filepath"].map(Path)
    val_df["filepath"]   = val_df["filepath"].map(Path)
    test_df["filepath"]  = test_df["filepath"].map(Path)

    # Verify files still exist on disk (path may have moved)
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        broken = (~df["filepath"].map(lambda p: p.exists())).sum()
        if broken:
            print(f" {broken:,} {name} filepaths no longer exist on disk")
            print(", If you moved the dataset, re-run preprocessing to rebuild CSVs.")

    total = len(train_df) + len(val_df) + len(test_df)
    print(f"\n  Loaded pre-saved splits:")
    print(f"  {'Split':<8}  {'Samples':>8}  {'%':>6}")
    print(f"  {'─'*28}")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"  {name:<8}  {len(df):>8,}  {len(df)/total*100:>5.1f}%")

    return train_df, val_df, test_df


def build_loaders(train_df, val_df, hp, device):
    pin = torch.cuda.is_available()
    n_threads = min(8, os.cpu_count())

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])

    train_loader = DataLoader(
        SkinLesionDataset(train_df, train_tf),
        batch_size=hp["BATCH_SIZE"], shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=pin,
    )
    val_loader = DataLoader(
        SkinLesionDataset(val_df, eval_tf),
        batch_size=hp["BATCH_SIZE"], shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=pin,
    )

    train_pf = ThreadedPrefetcher(train_loader, n_threads, n_threads * 2, device, pin)
    val_pf   = ThreadedPrefetcher(val_loader, max(2, n_threads // 2),
                                  max(4, n_threads), device, pin)
    return train_pf, val_pf


# Training

def run():
    if config.BASE_PATH is None:
        raise RuntimeError("Paths not set. Run client.py first.")

    print("\n" + "=" * 60)
    print("  TRAINING")
    print("=" * 60)

    hp = prompt_hyperparams()
    if hp is None:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    print("\nLoading data …")
    train_df, val_df, _ = load_dataframes()

    # Exit if CSVs were missing
    if train_df is None:
        return

    train_pf, val_pf = build_loaders(train_df, val_df, hp, device)

    # Class weights
    num_cls = config.NUM_CLASSES
    raw     = train_df["label"].value_counts()
    counts  = (raw.reindex(range(num_cls), fill_value=0)
                  .sort_index().values.astype(float).clip(min=1.0))
    weights = torch.tensor(counts.sum() / (num_cls * counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_cls)
    model    = model.to(device)

    optimizer = AdamW(model.parameters(),
                      lr=hp["LR"], weight_decay=hp["WEIGHT_DECAY"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    print(f"\n  Model  : ResNet-18  →  {num_cls} classes")
    print(f"  Params : {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Training loop
    best_val_acc = 0.0
    history      = []

    for epoch in range(1, hp["NUM_EPOCHS"] + 1):
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch:>3d} / {hp['NUM_EPOCHS']}")
        print(f"{'='*60}")

        # Train
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        bar = tqdm(train_pf, total=len(train_pf), desc="  Train",
                   leave=False, unit="batch", ncols=70)
        for imgs, labels in bar:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            bs        = imgs.size(0)
            t_loss   += loss.item() * bs
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_total  += bs
            bar.set_postfix(loss=f"{loss.item():.4f}")

        t_loss /= t_total
        t_acc   = t_correct / t_total

        # Validate
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_pf, total=len(val_pf),
                                     desc="  Val  ", leave=False,
                                     unit="batch", ncols=70):
                imgs   = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(imgs)
                loss   = criterion(logits, labels)
                bs      = imgs.size(0)
                v_loss += loss.item() * bs
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total   += bs

        v_loss /= v_total
        v_acc   = v_correct / v_total

        scheduler.step(v_acc)
        lr = optimizer.param_groups[0]["lr"]

        history.append(dict(epoch=epoch, train_loss=t_loss, train_acc=t_acc,
                            val_loss=v_loss, val_acc=v_acc, lr=lr))

        print(f"  Train  loss: {t_loss:.4f}  acc: {t_acc:.4f}")
        print(f"  Val    loss: {v_loss:.4f}  acc: {v_acc:.4f}")
        print(f"  LR         : {lr:.2e}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "optim_state" : optimizer.state_dict(),
                "val_acc"     : v_acc,
                "class_names" : config.CLASS_NAMES,
                "drop_class"  : config.DROP_CLASS,
                "hyperparams" : hp,
            }, config.CKPT_PATH)
            print(f" New best val acc {v_acc:.4f} — checkpoint saved.")

    print(f"\n{'='*60}")
    print(f"  Training complete.  Best val acc: {best_val_acc:.4f}")
    print(f"  Checkpoint: {config.CKPT_PATH}")

    # Save training curves
    save_training_curves(history)


def save_training_curves(history):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        hist_df = pd.DataFrame(history)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        axes[0].plot(hist_df["epoch"], hist_df["train_loss"], label="Train")
        axes[0].plot(hist_df["epoch"], hist_df["val_loss"],   label="Val")
        axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(hist_df["epoch"], hist_df["train_acc"], label="Train")
        axes[1].plot(hist_df["epoch"], hist_df["val_acc"],   label="Val")
        axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(hist_df["epoch"], hist_df["lr"])
        axes[2].set_title("Learning Rate"); axes[2].set_xlabel("Epoch")
        axes[2].set_yscale("log"); axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        out = config.CKPT_PATH.parent / "training_curves.png"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  Training curves saved → {out}")
    except Exception as e:
        print(f"  (Could not save training curves: {e})")


if __name__ == "__main__":
    if config.BASE_PATH is None:
        path = input("Enter dataset base directory: ").strip()
        config.set_paths(Path(path))
    run()