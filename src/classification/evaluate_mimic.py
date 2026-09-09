import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from mimic_dataset import MimicDataset, PATHOLOGY_LABELS
from model_mimic import get_mimic_model

def evaluate_and_test():
    # ── Configuration ─────────────────────────────────────────────
    root_dir = "C:/Users/ganne/OneDrive/Documents/DL/mimic/official_data_iccv_final"
    val_csv = "C:/Users/ganne/OneDrive/Documents/DL/training/mimic_cxr_aug_validate.csv"
    checkpoint_path = "C:/Users/ganne/OneDrive/Documents/DL/training/checkpoints_mimic/best_mimic_model.pth"
    num_classes = 14
    batch_size = 32
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    # Same validation transforms as training script
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading Validation Dataset...")
    val_dataset = MimicDataset(root_dir=root_dir, labels_csv=val_csv, transform=val_transform)
    
    if len(val_dataset) == 0:
        print("No validation data found!")
        return

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("\nLoading Model...")
    model = get_mimic_model(num_classes)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model weights not found at {checkpoint_path}")
        return
        
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!\n")

    # ── 1. Single Image Test ──────────────────────────────────────
    print("=" * 60)
    print("          TESTING ON A SINGLE RANDOM IMAGE")
    print("=" * 60)
    
    # Pick a random index
    idx = random.randint(0, len(val_dataset) - 1)
    # The dataset returns (transformed_image, label_tensor)
    image_tensor, true_labels = val_dataset[idx]
    
    # Move to device and add batch dimension (C, H, W) -> (1, C, H, W)
    image_input = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_input)
        # Apply sigmoid to convert logits to probabilities [0, 1]
        probs = torch.sigmoid(outputs).squeeze(0).cpu().numpy()
        
    true_labels_np = true_labels.numpy()
    
    print(f"\nResults for Validation Image #{idx}:")
    print(f"{'Pathology':<30} | {'Probability':<12} | {'Ground Truth'}")
    print("-" * 60)
    for i, label_name in enumerate(PATHOLOGY_LABELS):
        prob = probs[i]
        truth = int(true_labels_np[i])
        marker = "⭐ DETECTED" if prob > 0.5 else ""
        print(f"{label_name:<30} | {prob*100:5.1f}%      | {truth}  {marker}")

    # ── 2. Full Validation Accuracy Check ──────────────────────────
    print("\n" + "=" * 60)
    print("          CALCULATING FULL VALIDATION ACCURACY")
    print("=" * 60)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())
            
    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Threshold at 0.5 for discrete predictions
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Calculate global metrics
    global_acc = accuracy_score(all_targets, binary_preds)
    global_micro_f1 = f1_score(all_targets, binary_preds, average='micro', zero_division=0)
    
    print(f"\nOverall Exact Match Accuracy (All labels correct): {global_acc * 100:.2f}%")
    print(f"Overall Micro F1-Score: {global_micro_f1:.4f}\n")
    
    print(f"{'Pathology':<30} | {'Class Acc':<10} | {'AUC-ROC'}")
    print("-" * 60)
    
    for i, target_name in enumerate(PATHOLOGY_LABELS):
        class_targets = all_targets[:, i]
        class_probs = all_preds[:, i]
        class_preds = binary_preds[:, i]
        
        # Accuracy per class
        class_acc = accuracy_score(class_targets, class_preds)
        
        # AUROC per class (only calculate if both positive and negative samples exist in val set)
        if len(np.unique(class_targets)) > 1:
            auc = roc_auc_score(class_targets, class_probs)
            auc_str = f"{auc:.3f}"
        else:
            auc_str = "N/A (No Pos)"
            
        print(f"{target_name:<30} | {class_acc*100:6.1f}%   | {auc_str}")

if __name__ == "__main__":
    evaluate_and_test()
