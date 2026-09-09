import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.append("C:/Users/ganne/OneDrive/Documents/DL/training")
from mimic_dataset import MimicDataset
from model_mimic import get_mimic_model

# ─── Config ───────────────────────────────────────────────────────────────────
ROOT_DIR    = "C:/Users/ganne/OneDrive/Documents/DL/mimic/official_data_iccv_final"
TRAIN_CSV   = "C:/Users/ganne/OneDrive/Documents/DL/training/mimic_cxr_aug_train.csv"
VAL_CSV     = "C:/Users/ganne/OneDrive/Documents/DL/training/mimic_cxr_aug_validate.csv"
RESUME_FROM = "checkpoints_mimic/best_mimic_model.pth"   # p19 best model
SAVE_BEST   = "checkpoints_mimic/best_p16_model.pth"
CKPT_DIR    = "checkpoints_mimic"

PATIENT_PREFIX = "16"   # filters only p16 patients from the CSV
BATCH_SIZE     = 32
EPOCHS         = 10
LR             = 0.001
NUM_CLASSES    = 14

# ─── Main ─────────────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), "CUDA GPU required!"
device = torch.device("cuda")
print(f"Using device: {device}\n")

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading p16 training dataset...")
train_ds = MimicDataset(ROOT_DIR, TRAIN_CSV, train_transform, patient_prefix=PATIENT_PREFIX)
print("Loading p16 validation dataset...")
val_ds   = MimicDataset(ROOT_DIR, VAL_CSV,   val_transform,   patient_prefix=PATIENT_PREFIX)

if len(train_ds) == 0:
    print("ERROR: No training images found for p16. Check ROOT_DIR and CSV paths.")
    sys.exit(1)

print(f"\nTraining on {len(train_ds)} images, validating on {len(val_ds)} images.\n")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ─── Model ────────────────────────────────────────────────────────────────────
model = get_mimic_model(NUM_CLASSES).to(device)

if os.path.exists(RESUME_FROM):
    print(f"[INCREMENTAL] Resuming from: {RESUME_FROM}")
    ckpt = torch.load(RESUME_FROM, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    print("✅ Loaded p19 knowledge successfully!\n")
else:
    print(f"[WARN] Checkpoint not found at {RESUME_FROM}. Training from scratch.\n")

from focal_loss import FocalLoss
criterion = FocalLoss(alpha=0.25, gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
scaler    = torch.amp.GradScaler('cuda')

os.makedirs(CKPT_DIR, exist_ok=True)
best_val_loss = float('inf')

# ─── Training Loop ────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    # Train
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{running_loss / (pbar.n + 1):.4f}"})
    train_loss = running_loss / len(train_loader)

    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        vpbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  ")
        for images, labels in vpbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss    = criterion(outputs, labels)
            val_loss += loss.item()
            vpbar.set_postfix({'loss': f"{val_loss / (vpbar.n + 1):.4f}"})
    val_loss /= len(val_loader)

    print(f"\nEpoch {epoch+1} Summary — Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    scheduler.step(val_loss)

    # Save epoch checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, f"{CKPT_DIR}/p16_epoch_{epoch}.pth")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), SAVE_BEST)
        print(f"⭐ New best val loss: {best_val_loss:.4f} → saved to {SAVE_BEST}\n")
    else:
        print()

print(f"\n✅ p16 training complete. Best val loss: {best_val_loss:.4f}")
print(f"   Best model saved to: {SAVE_BEST}")
