import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from mimic_dataset import MimicDataset
from model_mimic import get_mimic_model
from tqdm import tqdm

def train_mimic(root_dir, train_csv, val_csv, resume_path=None, batch_size=32, epochs=10, learning_rate=0.001):
    # Parameters
    num_classes = 14 # Standard medical pathologies
    
    assert torch.cuda.is_available(), "GPU is critically required per the user's request. CUDA is not available!"
    device = torch.device("cuda")
    print(f"Using exactly device: {device}")

    # Transformations (Added minor data augmentation for training robustness)
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

    # Dataset & Dataloader
    print(f"Loading Training Dataset from {train_csv}...")
    train_dataset = MimicDataset(root_dir=root_dir, labels_csv=train_csv, transform=train_transform)
    
    print(f"Loading Validation Dataset from {val_csv}...")
    val_dataset = MimicDataset(root_dir=root_dir, labels_csv=val_csv, transform=val_transform)
    
    if len(train_dataset) == 0:
        print("Error: No images found for training.")
        return
        
    # Windows sometimes throws errors with multiple workers on custom loaders, keeping it at 0 to be safe
    # but using pin_memory for speed.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Model, Loss, Optimizer
    model = get_mimic_model(num_classes).to(device)
    
    if resume_path and os.path.exists(resume_path):
        print(f"\n[INCREMENTAL TRAINING] Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Successfully loaded previous knowledge!\n")
    
    from focal_loss import FocalLoss
    # Focal Loss auto-balances the extreme class imbalances (e.g. Rare diseases like Pulmonary Fibrosis)
    criterion = FocalLoss(alpha=0.25, gamma=2.0) 
    
    # Adding L2 Regularization (weight_decay) to prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Adaptive Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # Mixed precision scaler (massively speeds up training on modern GPUs)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_loss = float('inf')
    os.makedirs("checkpoints_mimic", exist_ok=True)

    # Training Loop
    print(f"Starting training on {len(train_dataset)} images, validating on {len(val_dataset)} images...")
    for epoch in range(epochs):
        # ────────────── TRAIN ──────────────
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Using Mixed Precision (AMP)
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{running_loss / (pbar.n + 1):.4f}"})

        train_loss = running_loss / len(train_loader)
        
        # ────────────── VALIDATION ──────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, labels in val_pbar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f"{val_loss / (val_pbar.n + 1):.4f}"})
                
        val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1} Summary - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Step LR Scheduler
        scheduler.step(val_loss)

        # Save checkpoint with metadata
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, f"checkpoints_mimic/mimic_epoch_{epoch}.pth")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"⭐ New best validation loss: {best_val_loss:.4f}. Saving best model...\n")
            torch.save(model.state_dict(), "checkpoints_mimic/best_mimic_model.pth")
        else:
            print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--root_dir", type=str, default="C:/Users/ganne/OneDrive/Documents/DL/mimic/official_data_iccv_final", help="Path to image folder")
    parser.add_argument("--train_csv", type=str, default="C:/Users/ganne/OneDrive/Documents/DL/training/mimic_cxr_aug_train.csv", help="Path to training CSV")
    parser.add_argument("--val_csv", type=str, default="C:/Users/ganne/OneDrive/Documents/DL/training/mimic_cxr_aug_validate.csv", help="Path to validation CSV")
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    
    args = parser.parse_args()
    
    train_mimic(
        root_dir=args.root_dir, 
        train_csv=args.train_csv, 
        val_csv=args.val_csv, 
        resume_path=args.resume,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr
    )
