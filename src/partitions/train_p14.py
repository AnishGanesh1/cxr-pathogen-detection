import os
import sys
import glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.append("C:/Users/ganne/OneDrive/Documents/DL/training")
from mimic_dataset import MimicDataset
from model_mimic import get_mimic_model

# ─── Config ───────────────────────────────────────────────────────────────────
# Note: p14 data is located in C:/Users/ganne/OneDrive/Documents/DL/training/p14
# The fallback in MimicDataset will resolve "files/p14/..." to "ROOT_DIR/p14/..."
ROOT_DIR    = "C:/Users/ganne/OneDrive/Documents/DL/training"
TRAIN_CSV   = "C:/Users/ganne/OneDrive/Documents/DL/training/mimic_cxr_aug_train.csv"
VAL_CSV     = "C:/Users/ganne/OneDrive/Documents/DL/training/mimic_cxr_aug_validate.csv"
RESUME_FROM = "checkpoints_mimic/best_p18_model.pth"
SAVE_BEST   = "checkpoints_mimic/best_p14_model.pth"
CKPT_DIR    = "checkpoints_mimic"

PATIENT_PREFIX = "14"   # filters only p14 patients
BATCH_SIZE     = 32
EPOCHS         = 10
LR             = 0.001
NUM_CLASSES    = 14

PATHOLOGY_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

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

print(f"Loading p{PATIENT_PREFIX} training dataset from {ROOT_DIR}...")
train_ds = MimicDataset(ROOT_DIR, TRAIN_CSV, train_transform, patient_prefix=PATIENT_PREFIX)
print(f"Loading p{PATIENT_PREFIX} validation dataset...")
val_ds   = MimicDataset(ROOT_DIR, VAL_CSV,   val_transform,   patient_prefix=PATIENT_PREFIX)

if len(train_ds) == 0:
    print(f"ERROR: No training images found for p{PATIENT_PREFIX}. Check ROOT_DIR and CSV paths.")
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
    print("✅ Loaded p18 knowledge successfully!\n")
else:
    print(f"[WARN] Checkpoint not found at {RESUME_FROM}. Training from scratch.\n")

# To prevent error in gradcam later, we enable gradients for the whole model once
for param in model.parameters():
    param.requires_grad = True

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
    }, f"{CKPT_DIR}/p{PATIENT_PREFIX}_epoch_{epoch}.pth")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), SAVE_BEST)
        print(f"⭐ New best val loss: {best_val_loss:.4f} → saved to {SAVE_BEST}\n")
    else:
        print()

print(f"\n✅ p{PATIENT_PREFIX} training complete. Best val loss: {best_val_loss:.4f}")
print(f"   Best model saved to: {SAVE_BEST}")

# ─── Post-Training Visual Inference ──────────────────────────────────────────
print("\n" + "="*60)
print(f"  RUNNING VISUAL INFERENCE ON p{PATIENT_PREFIX} SAMPLE")
print("="*60)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Find a sample image for p14 (data is in ROOT_DIR/p14)
imgs = glob.glob(f"{ROOT_DIR}/p{PATIENT_PREFIX}/**/*.jpg", recursive=True)
if not imgs:
    print(f"[WARN] No p{PATIENT_PREFIX} sample images found for inference.")
    sys.exit(0)
sample_img_path = imgs[0]
output_png = f"inference_output_p{PATIENT_PREFIX}.png"

print(f"Sample Image: {sample_img_path}")

# Load best model for inference
model.load_state_dict(torch.load(SAVE_BEST, map_location=device))
model.eval()

image = Image.open(sample_img_path).convert("RGB")
image_tensor = val_transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()

results = sorted(zip(PATHOLOGY_LABELS, probabilities), key=lambda x: x[1], reverse=True)

# Grad-CAM
has_gradcam = False
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    has_gradcam = True
except ImportError:
    print("[WARN] pytorch-grad-cam not installed. Skipping CAM visualization.")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle(f"MIMIC-CXR Pathology Detection — p{PATIENT_PREFIX}", fontsize=14, fontweight='bold')

vis_image = np.array(image).astype(np.float32) / 255.0

top_idx = -1
for i, (pathology, prob) in enumerate(zip(PATHOLOGY_LABELS, probabilities)):
    if pathology == results[0][0] and prob > 0.3:
        top_idx = i
        break

if has_gradcam and top_idx != -1:
    target_layers = [model.features[-1]]
    # Ensure requires_grad is True for CAM
    for param in model.parameters(): param.requires_grad = True
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(top_idx)]
    g_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]
    vis_resized = cv2.resize(vis_image, (256, 256))
    visualization = show_cam_on_image(vis_resized, g_cam, use_rgb=True)
    axes[0].imshow(visualization)
    axes[0].set_title(f"Grad-CAM: {results[0][0]}", fontsize=12)
else:
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12)
axes[0].axis('off')

# Detection list
axes[1].axis('off')
y = 0.95
axes[1].text(0.05, y, f"Pathology Detections (p{PATIENT_PREFIX}):", fontsize=15, fontweight='bold', transform=axes[1].transAxes)
y -= 0.1
detected_any = False
for pathology, prob in results:
    if prob > 0.3:
        axes[1].text(0.08, y, f"{pathology}: {prob*100:.1f}%", transform=axes[1].transAxes, 
                     fontsize=13, fontweight='bold', color='red',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', pad=4))
        y -= 0.08
        detected_any = True

if not detected_any:
    axes[1].text(0.08, y, "No findings > 30%", transform=axes[1].transAxes, fontsize=13, color='green')

plt.tight_layout()
plt.savefig(output_png, bbox_inches='tight', dpi=150)
plt.close()

print(f"✅ Visual output saved to: {output_png}")
