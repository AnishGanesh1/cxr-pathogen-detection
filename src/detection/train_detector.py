import math
import os
import sys
import time
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

try:
    import utils
    from dataset import VinBigDataDataset
    from model import get_model
    from evaluate import evaluate
except ImportError:
    from src import utils
    from src.dataset import VinBigDataDataset
    from src.model import get_model
    from src.evaluate import evaluate

TRAIN_IMG_DIR = r"C:\Users\ganne\OneDrive\Documents\DL\mimic\official_data_iccv_final\files\p10"
TRAIN_ANN_FILE = r"C:\Users\ganne\OneDrive\Documents\DL\mimic\instance_train.json"

NUM_CLASSES = 15

BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
STEP_SIZE = 3
GAMMA = 0.1
NUM_WORKERS = 2
CHECKPOINT_DIR = r"D:\checkpoints_mimic"

VAL_SPLIT = 0.1
SEED = 42
LOG_EVERY = 50


def get_transform():
    return T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
    ])


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    losses_per_iter = []
    header = f"Epoch [{epoch}]"

    # Warm-up scheduler for first epoch
    warmup_scheduler = None
    if epoch == 0:
        warmup_iters = min(1000, max(1, len(data_loader) - 1))
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0 / 1000, total_iters=warmup_iters
        )

    start_time = time.time()
    seen_images = 0

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        seen_images += len(images)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_val = losses.item()

        if not math.isfinite(loss_val):
            print(f"Non-finite loss: {loss_val}. Stopping.")
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if warmup_scheduler is not None:
            warmup_scheduler.step()

        losses_per_iter.append(loss_val)

        if (i + 1) % LOG_EVERY == 0 or (i + 1) == len(data_loader):
            avg_loss = sum(losses_per_iter) / len(losses_per_iter)
            elapsed = max(1e-8, time.time() - start_time)
            ips = seen_images / elapsed
            print(
                f"  {header} step [{i+1}/{len(data_loader)}] "
                f"avg loss: {avg_loss:.4f} | {ips:.2f} img/s"
            )

    avg_loss = sum(losses_per_iter) / len(losses_per_iter)
    epoch_time = time.time() - start_time
    print(f"{header} finished | avg loss: {avg_loss:.4f} | time: {epoch_time/60:.2f} min")
    return avg_loss, epoch_time


def main():
    if not os.path.isdir(TRAIN_IMG_DIR):
        raise FileNotFoundError(f"Image directory not found: {TRAIN_IMG_DIR}")
    if not os.path.isfile(TRAIN_ANN_FILE):
        raise FileNotFoundError(f"Annotation file not found: {TRAIN_ANN_FILE}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    transform = get_transform()

    full_dataset = VinBigDataDataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE, transforms=transform)
    full_dataset_eval = VinBigDataDataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE, transforms=transform)

    if len(full_dataset) < 2:
        raise ValueError("Dataset is too small. Need at least 2 images for train/val split.")

    # Fixed split for reproducibility
    g = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(full_dataset), generator=g).tolist()

    split = int((1.0 - VAL_SPLIT) * len(full_dataset))
    split = max(1, min(split, len(full_dataset) - 1))

    train_idx = indices[:split]
    val_idx = indices[split:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset_eval, val_idx)

    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        collate_fn=utils.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        collate_fn=utils.collate_fn,
    )

    model = get_model(NUM_CLASSES)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    lr_sched = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE, gamma=GAMMA
    )

    total_train_time = 0.0

    for epoch in range(NUM_EPOCHS):
        avg_loss, epoch_time = train_one_epoch(model, optimizer, train_loader, device, epoch)
        total_train_time += epoch_time
        lr_sched.step()

        print(f"\nRunning validation for epoch {epoch}...")
        metrics = evaluate(model, val_loader, device, TRAIN_ANN_FILE)
        if metrics:
            print(f"  AP@[0.50:0.95] = {metrics.get('AP@[0.50:0.95]', 0):.4f}")
            print(f"  AP@0.50        = {metrics.get('AP@0.50', 0):.4f}")
        else:
            print("  No metrics returned by evaluator.")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": avg_loss,
                "metrics": metrics,
                "config": {
                    "train_img_dir": TRAIN_IMG_DIR,
                    "train_ann_file": TRAIN_ANN_FILE,
                    "num_classes": NUM_CLASSES,
                    "batch_size": BATCH_SIZE,
                    "num_epochs": NUM_EPOCHS,
                    "lr": LR,
                },
            },
            ckpt_path,
        )
        print(f"Checkpoint saved: {ckpt_path}\n")

    print(f"Training complete. Total train loop time: {total_train_time/3600:.2f} hours")


if __name__ == "__main__":
    main()
