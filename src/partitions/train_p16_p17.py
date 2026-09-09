"""
Train p16 then p17 incrementally from the existing best checkpoint,
then produce visual inference_output PNGs (like evaluate.py) for each.
"""
import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
sys.path.append("C:/Users/ganne/OneDrive/Documents/DL/training")
from mimic_dataset import MimicDataset
from model_mimic import get_mimic_model
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────
MIMIC_ROOT   = "C:/Users/ganne/OneDrive/Documents/DL/mimic/official_data_iccv_final"
TRAIN_CSV    = "C:/Users/ganne/OneDrive/Documents/DL/training/mimic_cxr_aug_train.csv"
VAL_CSV      = "C:/Users/ganne/OneDrive/Documents/DL/training/mimic_cxr_aug_validate.csv"
CKPT_DIR     = "checkpoints_mimic"

BATCH_SIZE   = 32
EPOCHS       = 10
LR           = 0.001
NUM_CLASSES  = 14

GROUPS = [
    # (name,  patient_prefix,  resume_checkpoint,                    save_best_as)
    ("p16", "16", f"{CKPT_DIR}/best_mimic_model.pth",  f"{CKPT_DIR}/best_p16_model.pth"),
    ("p17", "17", f"{CKPT_DIR}/best_p16_model.pth",    f"{CKPT_DIR}/best_p17_model.pth"),
]

PATHOLOGY_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

# ─── Transforms ───────────────────────────────────────────────────────────────
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

# ─── Training function ────────────────────────────────────────────────────────
def train_group(name, patient_prefix, resume_path, save_best_as):
    print(f"\n{'='*60}")
    print(f"  Training group: {name.upper()}  (prefix='{patient_prefix}')")
    print(f"  Resuming from : {resume_path}")
    print(f"{'='*60}\n")

    assert torch.cuda.is_available(), "CUDA GPU required"
    device = torch.device("cuda")

    print(f"Loading training dataset for {name}...")
    train_ds = MimicDataset(MIMIC_ROOT, TRAIN_CSV, train_transform, patient_prefix)
    print(f"Loading validation dataset for {name}...")
    val_ds   = MimicDataset(MIMIC_ROOT, VAL_CSV,   val_transform,   patient_prefix)

    if len(train_ds) == 0:
        print(f"[WARN] No training images found for {name}. Skipping.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = get_mimic_model(NUM_CLASSES).to(device)
    if resume_path and os.path.exists(resume_path):
        print(f"[INCREMENTAL] Loading weights from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print("✅ Previous knowledge loaded!\n")

    from focal_loss import FocalLoss
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    scaler    = torch.amp.GradScaler('cuda')

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # Train
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{name}] Epoch {epoch+1}/{EPOCHS} Train")
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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
            vpbar = tqdm(val_loader, desc=f"[{name}] Epoch {epoch+1}/{EPOCHS} Val")
            for images, labels in vpbar:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                val_loss += loss.item()
                vpbar.set_postfix({'loss': f"{val_loss / (vpbar.n + 1):.4f}"})
        val_loss /= len(val_loader)

        print(f"\n[{name}] Epoch {epoch+1} Summary — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        # Save per-epoch checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, f"{CKPT_DIR}/{name}_epoch_{epoch}.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"⭐ New best for {name}: {best_val_loss:.4f} → saved to {save_best_as}\n")
            torch.save(model.state_dict(), save_best_as)
        else:
            print()

    print(f"\n✅ Done training {name}. Best val loss: {best_val_loss:.4f}")
    return model

# ─── Visual Inference (same style as evaluate.py) ────────────────────────────
def run_visual_inference(name, checkpoint_path, image_path, output_png):
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    print(f"\n[{name}] Running visual inference on:\n  {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_mimic_model(num_classes=14)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image        = Image.open(image_path).convert("RGB")
    image_tensor = val_tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs       = model(image_tensor)
        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()

    results = sorted(zip(PATHOLOGY_LABELS, probabilities), key=lambda x: x[1], reverse=True)

    # Try GradCAM
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        has_gradcam = True
    except ImportError:
        has_gradcam = False
        print("  [WARN] pytorch-grad-cam not installed — skipping GradCAM")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"MIMIC-CXR Pathology Detection — {name.upper()}", fontsize=14, fontweight='bold')

    vis_image = np.array(image).astype(np.float32) / 255.0

    # Find top passing-threshold pathology index in original order
    top_idx = -1
    for i, (pathology, prob) in enumerate(zip(PATHOLOGY_LABELS, probabilities)):
        if pathology == results[0][0] and prob > 0.3:
            top_idx = i
            break

    if has_gradcam and top_idx != -1:
        import cv2
        target_layers = [model.features[-1]]
        model.eval()
        for param in model.parameters():
            param.requires_grad = True
        cam      = GradCAM(model=model, target_layers=target_layers)
        targets  = [ClassifierOutputTarget(top_idx)]
        g_cam    = cam(input_tensor=image_tensor, targets=targets)[0, :]
        vis_resized = cv2.resize(vis_image, (256, 256))
        visualization = show_cam_on_image(vis_resized, g_cam, use_rgb=True)
        axes[0].imshow(visualization)
        axes[0].set_title(f"Grad-CAM: {results[0][0]}", fontsize=12)
    else:
        axes[0].imshow(image)
        axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')

    # Detection panel
    axes[1].axis('off')
    y = 0.97
    axes[1].text(0.05, y, f"Pathology Detections ({name.upper()}):",
                 fontsize=15, fontweight='bold', transform=axes[1].transAxes, va='top')
    y -= 0.10

    print(f"\n{'─'*50}")
    print(f"=== {name.upper()} PATHOLOGY DETECTIONS:")
    print(f"{'─'*50}")
    detected_any = False
    for pathology, prob in results:
        marker = "[X]" if prob > 0.3 else "[ ]"
        print(f"{marker} {pathology:30s}: {prob * 100:.2f}%")
        if prob > 0.3:
            axes[1].text(0.08, y, f"{pathology}: {prob * 100:.1f}%",
                         transform=axes[1].transAxes, fontsize=13, fontweight='bold',
                         color='red', va='top',
                         bbox=dict(facecolor='white', alpha=0.85, edgecolor='red', pad=4))
            y -= 0.09
            detected_any = True
    if not detected_any:
        axes[1].text(0.08, y, "No findings > 30%",
                     transform=axes[1].transAxes, fontsize=13, fontweight='bold',
                     color='green', va='top',
                     bbox=dict(facecolor='white', alpha=0.85, edgecolor='green', pad=4))
    print(f"{'─'*50}")

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\n✅ Visual output saved to: {output_png}")

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1: Train p16 and p17 sequentially
    for (name, prefix, resume, save_as) in GROUPS:
        train_group(name, prefix, resume, save_as)

    # Step 2: Find one sample image per group and run visual inference
    print("\n" + "="*60)
    print("  POST-TRAINING VISUAL INFERENCE")
    print("="*60)

    sample_images = {
        "p16": None,
        "p17": None,
    }
    for grp in ["p16", "p17"]:
        imgs = glob.glob(
            f"C:/Users/ganne/OneDrive/Documents/DL/mimic/official_data_iccv_final/files/{grp}/**/*.jpg",
            recursive=True
        )
        if imgs:
            sample_images[grp] = imgs[0]
        else:
            print(f"[WARN] No .jpg images found for {grp}")

    inference_map = {
        "p16": (f"{CKPT_DIR}/best_p16_model.pth", "inference_output_p16.png"),
        "p17": (f"{CKPT_DIR}/best_p17_model.pth", "inference_output_p17.png"),
    }
    for grp, (ckpt, out_png) in inference_map.items():
        img = sample_images.get(grp)
        if img and os.path.exists(ckpt):
            run_visual_inference(grp, ckpt, img, out_png)
        else:
            print(f"[SKIP] {grp}: checkpoint or image not found.")

    print("\n🎉 All done! Check inference_output_p16.png and inference_output_p17.png")
