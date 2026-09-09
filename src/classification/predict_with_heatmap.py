import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from PIL import Image

from mimic_dataset import MimicDataset, PATHOLOGY_LABELS
from model_mimic import get_mimic_model

class GradCAM:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM)
    for a given model and targeting a specific layer.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to extract the activations and gradients
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] contains the gradient w.r.t. the output of the layer
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        # Set model to evaluation mode
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Zero any existing gradients before backward pass
        self.model.zero_grad()
        
        # We want the gradient for the specific target class
        target = output[:, class_idx]
        target.backward(retain_graph=True)
        
        # Get extracted gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global Average Pooling on gradients across spatial dimensions
        weights = np.mean(gradients, axis=(1, 2))
        
        # Multiply each activation channel by its corresponding weight
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU to keep only features that have a positive influence on the class
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam, output

def visualize_heatmap_with_bbox():
    root_dir = "C:/Users/ganne/OneDrive/Documents/DL/mimic/official_data_iccv_final"
    val_csv = "C:/Users/ganne/OneDrive/Documents/DL/training/mimic_cxr_aug_validate.csv"
    checkpoint_path = "C:/Users/ganne/OneDrive/Documents/DL/training/checkpoints_mimic/best_mimic_model.pth"
    out_img_path = "C:/Users/ganne/OneDrive/Documents/DL/training/prediction_heatmap.png"

    # Minimal validations
    if not os.path.exists(val_csv):
        print(f"Error: Validation CSV not found at {val_csv}")
        return

    # Image Transform
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading Dataset...")
    val_dataset = MimicDataset(root_dir=root_dir, labels_csv=val_csv, transform=val_transform)
    
    if len(val_dataset) == 0:
        print("Dataset is empty. Exiting.")
        return
        
    print("Loading Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_mimic_model(14)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("Model checkpoint loaded.")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using untrained model (results will be noise).")

    model.to(device)

    # Attach GradCAM to the final layer of DenseNet-121
    # For DenseNet, 'features.norm5' is the final BatchNormalization 
    # immediately following the last dense block.
    target_layer = model.features.norm5
    grad_cam = GradCAM(model, target_layer)

    print("Running Inference (Searching for clear pathology)...")
    model.eval()
    while True:
        idx = random.randint(0, len(val_dataset) - 1)
        img_path, true_tensor = val_dataset.samples[idx]
        
        raw_img_pil = Image.open(img_path).convert("RGB")
        raw_img = np.array(raw_img_pil)
        img_tensor = val_transform(raw_img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            test_output = model(img_tensor)
            probs = torch.sigmoid(test_output).squeeze(0).cpu().numpy()
            
        target_class_idx = np.argmax(probs)
        if PATHOLOGY_LABELS[target_class_idx] == "Pleural Effusion":
            print(f"Selected image: {img_path}")
            break
    
    # We will compute the heatmap for the pathology the model is MOST confident about.
    # Alternatively, you could change this to a specific pathology index if desired.
    target_class_idx = np.argmax(probs)
    pathology_name = PATHOLOGY_LABELS[target_class_idx]
    predicted_prob = probs[target_class_idx] * 100

    print(f"Model predicts '{pathology_name}' with probability {predicted_prob:.1f}%")
    print(f"Generating heatmap for '{pathology_name}'...")

    # Generate Heatmap (this runs a backward pass, so we don't use torch.no_grad())
    cam, _ = grad_cam.generate_heatmap(img_tensor, target_class_idx)
    
    # Resize the low-resolution heatmap back to the original image dimensions
    cam_resized = cv2.resize(cam, (raw_img.shape[1], raw_img.shape[0]))
    
    # --- Generate Bounding Box from Heatmap ---
    # 1. Threshold the heatmap (keep only the top X% hottest pixels)
    # Using 0.6 * max (top 40%) usually gives a tight box around the primary lesion. 
    # Change threshold below to make box bigger/smaller
    threshold = 0.6 * np.max(cam_resized)
    binary_mask = (cam_resized > threshold).astype(np.uint8) * 255
    
    # 2. Find contours (connected components) of the thresholded mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Create visual output overlapping Original Image, Heatmap, and Bounding Box
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) # OpenCV uses BGR natively
    
    # Combine original image and colored heatmap
    superimposed_img = np.float32(heatmap_colored) * 0.4 + np.float32(raw_img) * 0.6
    superimposed_img = superimposed_img / np.max(superimposed_img)
    superimposed_img = np.uint8(255 * superimposed_img)

    # 4. Draw bounding boxes
    if contours:
        # We'll just draw a bounding box around the single LARGEST activated area
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Only draw if the box isn't impossibly tiny (filters noise)
        if w * h > 400: 
            cv2.rectangle(superimposed_img, (x, y), (x+w, y+h), (255, 0, 0), 3) # Pure Red box
            
            # Put label and probability slightly above the box
            label_text = f"{pathology_name}: {predicted_prob:.1f}%"
            cv2.putText(superimposed_img, label_text, (x, max(y - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # --- Plotting ---
    plt.figure(figsize=(16, 6))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(raw_img)
    plt.title("Original Medical Image")
    plt.axis('off')
    
    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(cam_resized, cmap='jet')
    plt.title(f"Grad-CAM Heatmap\n(Activation for '{pathology_name}')")
    plt.axis('off')
    
    # Bounding Box over Heatmap on Image
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title(f"Weakly Supervised Bounding Box\n(Threshold > 60% activation)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(out_img_path, dpi=200)
    print(f"\n✅ Success! Saved detailed prediction plot with bounding box to:\n{out_img_path}")

if __name__ == "__main__":
    visualize_heatmap_with_bbox()
