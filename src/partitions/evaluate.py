import sys
import torch
from torchvision import transforms
from PIL import Image
from model_mimic import get_mimic_model

# 14 standard CheXpert/MIMIC-CXR pathology labels
PATHOLOGY_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

def evaluate_image(image_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Needs to match the validation transform EXACTLY
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print(f"Loading image from: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Failed to open image: {e}")
        return
        
    image_tensor = val_transform(image).unsqueeze(0).to(device)
    
    print(f"Initializing model and loading weights from {checkpoint_path}...")
    model = get_mimic_model(num_classes=14)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("\nRunning Inference...")
    with torch.no_grad():
        # The model outputs raw logits since BCEWithLogitsLoss / FocalLoss was used
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
        
    results = sorted(zip(PATHOLOGY_LABELS, probabilities), key=lambda x: x[1], reverse=True)
    
    print("-" * 50)
    print("=== MIMIC-CXR PATHOLOGY DETECTIONS:")
    print("-" * 50)
    
    import matplotlib.pyplot as plt
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        has_gradcam = True
    except ImportError:
        print("pytorch-grad-cam not found. Skipping GradCAM.")
        has_gradcam = False
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Identify top pathology passing threshold
    top_pathology_idx = -1
    for i, (pathology, prob) in enumerate(zip(PATHOLOGY_LABELS, probabilities)):
        if pathology == results[0][0] and prob > 0.3:
            top_pathology_idx = i
            break
            
    vis_image = np.array(image).astype(np.float32) / 255.0

    if has_gradcam and top_pathology_idx != -1:
        # Use the last feature extraction layer of DenseNet
        target_layers = [model.features[-1]]
        
        # We need to temporarily enable grad for GradCAM
        model.eval()
        for param in model.parameters():
            param.requires_grad = True
            
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(top_pathology_idx)]
        
        # We must make sure input_tensor requires grad is not strictly needed by the new library, 
        # but we pass it anyway.
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Resize original image to 256x256 to match the transform if needed, 
        # but show_cam_on_image needs the image and cam to be the same size.
        # grayscale_cam is the size of image_tensor (256x256).
        # We should use the transformed image or resize vis_image to 256x256.
        import cv2
        vis_image_resized = cv2.resize(vis_image, (256, 256))
        
        visualization = show_cam_on_image(vis_image_resized, grayscale_cam, use_rgb=True)
        axes[0].imshow(visualization)
        axes[0].set_title(f"Grad-CAM: {results[0][0]}")
    else:
        axes[0].imshow(image)
        axes[0].set_title("Original Image (No high-confidence detections for CAM)")
        
    axes[0].axis('off')
    
    # Plot text on the second axis (outside the image)
    axes[1].axis('off')
    y_offset = 0.95
    detected_any = False
    
    axes[1].text(0.05, y_offset, "Pathology Detections:", fontsize=16, fontweight='bold', transform=axes[1].transAxes)
    y_offset -= 0.1
    
    for pathology, prob in results:
        marker = "[X]" if prob > 0.3 else "[ ]"
        print(f"{marker} {pathology:30s} : {prob * 100:.2f}%")
        
        if prob > 0.3:
            text = f"{pathology}: {prob * 100:.1f}%"
            axes[1].text(0.1, y_offset, text, transform=axes[1].transAxes, fontsize=14, fontweight='bold',
                    color='red', bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', pad=4))
            y_offset -= 0.08
            detected_any = True
            
    if not detected_any:
        axes[1].text(0.1, y_offset, "No findings > 30%", transform=axes[1].transAxes, fontsize=14, fontweight='bold',
                color='green', bbox=dict(facecolor='white', alpha=0.8, edgecolor='green', pad=4))

    print("-" * 50)
    
    output_path = "inference_output.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved visual output to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <path_to_image>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    ckpt_path = "checkpoints_mimic/best_mimic_model.pth"
    evaluate_image(img_path, ckpt_path)
