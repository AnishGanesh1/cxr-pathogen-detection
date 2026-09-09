import os
import ast
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image

# 14 standard CheXpert/MIMIC-CXR pathology labels
PATHOLOGY_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

# Keyword mapping for each pathology (case-insensitive search in report text)
KEYWORDS = {
    "Atelectasis":               ["atelectasis", "atelectatic", "collapse", "collapsed"],
    "Cardiomegaly":              ["cardiomegaly", "cardiac enlargement", "enlarged heart", "cardiomegalia"],
    "Consolidation":             ["consolidation", "consolidative", "airspace disease"],
    "Edema":                     ["edema", "oedema", "pulmonary edema", "vascular congestion"],
    "Enlarged Cardiomediastinum":["enlarged cardiomediastinum", "widened mediastinum", "mediastinal widening"],
    "Fracture":                  ["fracture", "rib fracture", "osseous", "broken rib"],
    "Lung Lesion":               ["lung lesion", "lung nodule", "nodule", "mass", "lesion"],
    "Lung Opacity":              ["opacity", "opacities", "opacification", "haziness"],
    "No Finding":                ["no acute", "no focal", "normal", "unremarkable", "no significant", "no evidence"],
    "Pleural Effusion":          ["pleural effusion", "effusion", "pleural fluid"],
    "Pleural Other":             ["pleural thickening", "pleural disease", "pleural abnormality"],
    "Pneumonia":                 ["pneumonia", "infection", "infectious"],
    "Pneumothorax":              ["pneumothorax", "pneumothoraces"],
    "Support Devices":           ["support device", "tube", "catheter", "pacemaker", "lead", "device"],
}


def _extract_labels_from_text(report_text: str):
    """Return a float tensor of shape [14] with binary labels from free-text report."""
    text_lower = report_text.lower()
    label_vec = []
    for pathology in PATHOLOGY_LABELS:
        found = any(kw in text_lower for kw in KEYWORDS[pathology])
        label_vec.append(1.0 if found else 0.0)
    return torch.tensor(label_vec, dtype=torch.float32)


def _parse_list_field(field: str):
    """Parse a CSV field that is a Python list literal, e.g. \"['a', 'b']\"."""
    try:
        result = ast.literal_eval(field)
        if isinstance(result, list):
            return result
        return [str(result)]
    except Exception:
        return [field]


class MimicDataset(Dataset):
    """
    Dataset for MIMIC-CXR augmented CSV.

    CSV columns expected:
        image       – Python list literal of relative image paths (relative to root_dir)
        text        – Python list literal of radiology report strings
        (AP, PA, Lateral columns are ignored; we use the flat `image` list)

    Labels are extracted from the concatenated report text using keyword matching
    against the 14 CheXpert pathology categories.
    """

    def __init__(self, root_dir: str, labels_csv: str, transform=None, patient_prefix=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # list of (image_path, label_tensor)

        with open(labels_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if patient_prefix is not None:
                    # Filter by subject ID starting with the given prefix (e.g., '19' for p19)
                    subject_id = str(row.get("subject_id", ""))
                    if not subject_id.startswith(patient_prefix):
                        continue
                # ── Images ────────────────────────────────────────────────
                image_paths = _parse_list_field(row["image"])

                # ── Labels from report text ────────────────────────────────
                text_list = _parse_list_field(row.get("text", "[]"))
                combined_text = " ".join(text_list)
                label_tensor = _extract_labels_from_text(combined_text)

                # ── One sample per image in this study ────────────────────
                for rel_path in image_paths:
                    abs_path = os.path.join(root_dir, rel_path)
                    
                    # Fallback: if 'files/p19/...' fails, try stripping 'files/' in case
                    # the data is extracted directly like root/p19/...
                    if not os.path.isfile(abs_path) and rel_path.startswith("files/"):
                        alt_path = os.path.join(root_dir, rel_path[len("files/"):])
                        if os.path.isfile(alt_path):
                            abs_path = alt_path

                    if os.path.isfile(abs_path):
                        self.samples.append((abs_path, label_tensor))

        print(f"MimicDataset: loaded {len(self.samples)} images from {labels_csv}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, labels
