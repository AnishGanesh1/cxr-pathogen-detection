# Transition to DenseNet-121 & Google Colab Workflow

This plan outlines the steps to upgrade the model to a DenseNet-121 architecture (better for X-rays) and sets up a workflow for training on Google Colab.

## User Review Required

> [!IMPORTANT]
> **Complete Retraining Required:** Moving to DenseNet-121 is a breaking change for your current weights. You must start training from scratch.
> 
> **Data Transfer to Colab:** You will need to upload your `mimic` dataset to Google Drive or a cloud-accessible location to use Colab effectively.

## Proposed Changes

---

### Phase 1: Architecture Upgrade (DenseNet-121)

#### [MODIFY] [model_mimic.py](file:///C:/Users/ganne/OneDrive/Documents/DL/training/model_mimic.py)
*   Replace ResNet-50 with DenseNet-121.
*   Update the linear head to match DenseNet's output features (1024 instead of 2048).

#### [MODIFY] [predict_with_heatmap.py](file:///C:/Users/ganne/OneDrive/Documents/DL/training/predict_with_heatmap.py)
*   Update Grad-CAM target layer to `model.features.norm5` (the standard final conv output for DenseNet-121).

---

### Phase 2: Portability & Colab Support

#### [MODIFY] [train_mimic.py](file:///C:/Users/ganne/OneDrive/Documents/DL/training/train_mimic.py)
*   Add command-line arguments for `root_dir`, `train_csv`, and `val_csv` so you don't have to edit the script every time you move between PC and Colab.

#### [NEW] [Colab_Notebook_Snippet.md](file:///C:/Users/ganne/.gemini/antigravity/brain/e37621bb-d647-491d-b88b-cee284e338d7/artifacts/Colab_Notebook_Snippet.md)
*   A ready-to-use Markdown file containing the cells you need to copy-paste into Colab to mount Drive, install dependencies, and run the training.

## Open Questions
1. **Dataset Size:** How many GB is your `official_data_iccv_final` folder? (This determines if we should unzip it inside Colab or stream it from Drive).

## Verification Plan

### Automated Tests
1. Instantiate the new DenseNet model locally and run one forward pass with a dummy tensor to ensure the head shape is correct.
2. Verify the Grad-CAM hook successfully captures gradients for the new `norm5` target layer.

### Manual Verification
1. Run `python train_mimic.py --epochs 1` locally to ensure the training loop starts correctly with the new architecture. 
