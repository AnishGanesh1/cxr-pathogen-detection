import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import sys
# Add training directory to path to import components
sys.path.append(r"c:\Users\ganne\OneDrive\Documents\DL\training")

from mimic_dataset import MimicDataset, PATHOLOGY_LABELS
from model_mimic import get_mimic_model

def evaluate_final():
    # ── Configuration ─────────────────────────────────────────────
    root_dir = r"c:\Users\ganne\OneDrive\Documents\DL\mimic\official_data_iccv_final"
    val_csv = r"c:\Users\ganne\OneDrive\Documents\DL\training\mimic_cxr_aug_validate.csv"
    checkpoint_path = r"c:\Users\ganne\OneDrive\Documents\DL\final_model\final_consolidated_model.pth"
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

    print("\nLoading Model Architecture...")
    model = get_mimic_model(num_classes)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model weights not found at {checkpoint_path}")
        return
        
    print(f"Loading Weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
        
    model.to(device)
    model.eval()
    print("Consolidated Model loaded successfully!\n")

    # ── Full Validation Accuracy Check ──────────────────────────
    print("=" * 60)
    print("          CONSISTENCY AND PERFORMANCE EVALUATION")
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
    
    print(f"\nConsolidated Model Accuracy (Exact Match): {global_acc * 100:.2f}%")
    print(f"Overall Micro F1-Score: {global_micro_f1:.4f}\n")
    
    print(f"{'Pathology':<30} | {'Class Acc':<10} | {'AUC-ROC'}")
    print("-" * 60)
    
    for i, target_name in enumerate(PATHOLOGY_LABELS):
        class_targets = all_targets[:, i]
        class_probs = all_preds[:, i]
        class_preds = binary_preds[:, i]
        
        class_acc = accuracy_score(class_targets, class_preds)
        
        if len(np.unique(class_targets)) > 1:
            auc = roc_auc_score(class_targets, class_probs)
            auc_str = f"{auc:.3f}"
        else:
            auc_str = "N/A"
            
        print(f"{target_name:<30} | {class_acc*100:6.1f}%   | {auc_str}")

if __name__ == "__main__":
    evaluate_final()
