# csv_to_coco.py
# Convert detection CSV -> COCO JSON (category_id is 0-based for this repo)

import os
import csv
import json
import argparse
from PIL import Image

def find_image_path(images_dir, image_id):
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = os.path.join(images_dir, f"{image_id}{ext}")
        if os.path.exists(p):
            return p, f"{image_id}{ext}"
    return None, None

def to_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV path")
    parser.add_argument("--images-dir", required=True, help="Directory containing image files")
    parser.add_argument("--output", required=True, help="Output COCO JSON path")
    args = parser.parse_args()

    # Expected CSV columns (VinBigData-like):
    # image_id, class_id, class_name, x_min, y_min, x_max, y_max
    # If your names differ, rename columns in CSV or edit below keys.
    image_id_col = "image_id"
    class_id_col = "class_id"
    class_name_col = "class_name"
    xmin_col, ymin_col, xmax_col, ymax_col = "x_min", "y_min", "x_max", "y_max"

    images = []
    annotations = []
    categories_map = {}  # raw class_id -> {"id": 0-based, "name": ...}
    image_id_map = {}    # image_id string -> numeric id
    image_size_cache = {}
    ann_id = 1

    with open(args.csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 1) Build image list
    next_img_id = 1
    unique_image_keys = sorted({r[image_id_col] for r in rows if r.get(image_id_col)})
    for img_key in unique_image_keys:
        img_path, file_name = find_image_path(args.images_dir, img_key)
        if img_path is None:
            print(f"[WARN] image not found for image_id={img_key}, skipping image")
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        image_id_map[img_key] = next_img_id
        image_size_cache[img_key] = (w, h)
        images.append({
            "id": next_img_id,
            "file_name": file_name,
            "width": w,
            "height": h
        })
        next_img_id += 1

    # 2) Build categories (0-based IDs)
    # Skip "No finding" by name.
    valid_rows = []
    for r in rows:
        cname = (r.get(class_name_col) or "").strip()
        if cname.lower() == "no finding":
            continue
        if r.get(image_id_col) not in image_id_map:
            continue

        raw_cid = r.get(class_id_col, "").strip()
        if raw_cid == "":
            continue

        if raw_cid not in categories_map:
            categories_map[raw_cid] = {
                "id": len(categories_map),  # 0-based
                "name": cname if cname else f"class_{raw_cid}"
            }
        valid_rows.append(r)

    categories = sorted(
        ({"id": v["id"], "name": v["name"]} for v in categories_map.values()),
        key=lambda x: x["id"]
    )

    # 3) Build annotations
    for r in valid_rows:
        img_key = r[image_id_col]
        w_img, h_img = image_size_cache[img_key]

        x_min = to_float(r.get(xmin_col))
        y_min = to_float(r.get(ymin_col))
        x_max = to_float(r.get(xmax_col))
        y_max = to_float(r.get(ymax_col))
        if None in (x_min, y_min, x_max, y_max):
            continue

        # clamp + sanitize
        x_min = max(0.0, min(x_min, w_img - 1))
        y_min = max(0.0, min(y_min, h_img - 1))
        x_max = max(0.0, min(x_max, w_img - 1))
        y_max = max(0.0, min(y_max, h_img - 1))

        bw = max(0.0, x_max - x_min)
        bh = max(0.0, y_max - y_min)
        if bw <= 0 or bh <= 0:
            continue

        raw_cid = r[class_id_col].strip()
        coco_cid = categories_map[raw_cid]["id"]  # 0-based

        annotations.append({
            "id": ann_id,
            "image_id": image_id_map[img_key],
            "category_id": coco_cid,
            "bbox": [x_min, y_min, bw, bh],
            "area": bw * bh,
            "iscrowd": 0
        })
        ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print(f"[OK] saved: {args.output}")
    print(f"images={len(images)} annotations={len(annotations)} categories={len(categories)}")

if __name__ == "__main__":
    main()
