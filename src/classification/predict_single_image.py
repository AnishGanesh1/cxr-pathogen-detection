import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from mimic_dataset import MimicDataset, PATHOLOGY_LABELS
from model_mimic import get_mimic_model

def visualize_prediction():
    root_dir = "C:/Users/ganne/OneDrive/Documents/DL/mimic/official_data_iccv_final"
    val_csv = "C:/Users/ganne/OneDrive/Documents/DL/training/mimic_cxr_aug_validate.csv"
    checkpoint_path = "C:/Users/ganne/OneDrive/Documents/DL/training/checkpoints_mimic/best_mimic_model.pth"
    out_img_path = "C:/Users/ganne/.gemini/antigravity/brain/c75fa59c-3978-4a55-b802-c131eb22bcf2/prediction_output.png"

    # Transform
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading Dataset...")
    val_dataset = MimicDataset(root_dir=root_dir, labels_csv=val_csv, transform=val_transform)
    print("Loading Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_mimic_model(14)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    print("Running Inference (Searching for clear pathology)...")
    while True:
        idx = random.randint(0, len(val_dataset) - 1)
        img_path, true_tensor = val_dataset.samples[idx]
        raw_img = Image.open(img_path).convert("RGB")
        img_tensor = val_transform(raw_img)

        with torch.no_grad():
            outputs = model(img_tensor.unsqueeze(0).to(device))
            probs = torch.sigmoid(outputs).squeeze(0).cpu().numpy()
            
        highest_prob_idx = np.argmax(probs)
        if PATHOLOGY_LABELS[highest_prob_idx] != "No Finding":
            print(f"Selected image with primary prediction: {PATHOLOGY_LABELS[highest_prob_idx]}")
            break

    print("Plotting results...")
    plt.figure(figsize=(14, 7))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(raw_img)
    plt.axis('off')
    plt.title("Input Medical Image")

    # Plot Probabilities
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(PATHOLOGY_LABELS))
    
    # Reverse order for better top-down readability
    y_pos = y_pos[::-1]
    
    colors = ['green' if t == 1 else 'gray' for t in true_tensor.numpy()]
    plt.barh(y_pos, probs * 100, align='center', alpha=0.8, color=colors)
    plt.yticks(y_pos, PATHOLOGY_LABELS)
    plt.xlabel('Predicted Probability (%)')
    plt.title('Model Predictions (Green = Ground Truth Positive)')
    plt.xlim(0, 100)
    
    for i, (prob, truth) in enumerate(zip(probs, true_tensor.numpy())):
        y_loc = y_pos[i]
        plt.text(prob * 100 + 2, y_loc, f"{prob*100:.1f}%", va='center')

    plt.tight_layout()
    plt.savefig(out_img_path, dpi=150)
    print(f"Saved prediction plot to {out_img_path}")

if __name__ == "__main__":
    visualize_prediction()
