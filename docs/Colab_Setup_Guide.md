# 🚀 Training on Google Colab Guide

To train this model on Google Colab, follow these steps to set up your environment and data.

## 1. Prepare your Data
Since the MIMIC dataset is large, the fastest way is to ZIP your image folder and upload it to Google Drive.

1.  **Zip your folder:** `official_data_iccv_final` -> `mimic_data.zip`
2.  **Upload to Drive:** Place `mimic_data.zip`, `mimic_cxr_aug_train.csv`, and `mimic_cxr_aug_validate.csv` in a folder on your Drive (e.g., `Colab_DL/`).
3.  **Upload Code:** Upload `train_mimic.py`, `mimic_dataset.py`, `model_mimic.py`, and `focal_loss.py`.

## 2. Colab Notebook Setup
Create a new Notebook in Colab and set **Runtime Type** to **GPU (T4 or A100)**.

### Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Unzip Data (Fastest Access)
Running from Drive directly is slow. Unzipping to the local Colab disk is 10x faster.
```python
!unzip -q "/content/drive/MyDrive/Colab_DL/mimic_data.zip" -d "/content/mimic_data"
```

### Cell 3: Install Dependencies
```python
!pip install tqdm pillow torchvision torch
```

### Cell 4: Start Training
Use the new command-line arguments we added to point to the Colab paths.
```python
!python train_mimic.py \
    --root_dir "/content/mimic_data" \
    --train_csv "/content/drive/MyDrive/Colab_DL/mimic_cxr_aug_train.csv" \
    --val_csv "/content/drive/MyDrive/Colab_DL/mimic_cxr_aug_validate.csv" \
    --batch_size 32 \
    --epochs 10 \
    --lr 0.001
```

## 3. Important Tips for Colab
*   **Checkpoints:** The script saves checkpoints to a local folder `checkpoints_mimic`. **Google Colab deletes local files when the session ends.** You should modify the script or use a symlink to save them back to Drive:
    ```python
    !mkdir -p /content/drive/MyDrive/Colab_DL/checkpoints
    !ln -s /content/drive/MyDrive/Colab_DL/checkpoints /content/checkpoints_mimic
    ```
*   **Disconnecting:** Use Colab Pro or keep the tab active to prevent the session from timing out during long training runs.
