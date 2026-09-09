import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou


class VinBigCXRCOCODataset(Dataset):
    """COCO-style VinBigData CXR dataset for torchvision detection models."""

    def __init__(self, ann_file: Path, image_dir: Path, category_id_to_label: Dict[int, int] | None = None):
        self.ann_file = Path(ann_file)
        self.image_dir = Path(image_dir)

        data = json.loads(self.ann_file.read_text(encoding="utf-8"))
        self.images = data["images"]
        self.annotations = data["annotations"]
        self.categories = data["categories"]

        self.id_to_image = {img["id"]: img for img in self.images}
        self.image_ids = sorted(self.id_to_image.keys())

        anns_by_image = defaultdict(list)
        for ann in self.annotations:
            anns_by_image[ann["image_id"]].append(ann)
        self.anns_by_image = anns_by_image

        sorted_category_ids = sorted(c["id"] for c in self.categories)
        if category_id_to_label is None:
            # Label 0 is reserved for background in Faster R-CNN.
            self.category_id_to_label = {cid: i + 1 for i, cid in enumerate(sorted_category_ids)}
        else:
            self.category_id_to_label = category_id_to_label

        self.label_to_category_id = {v: k for k, v in self.category_id_to_label.items()}
        self.category_id_to_name = {c["id"]: c["name"] for c in self.categories}

        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_info = self.id_to_image[image_id]
        image_path = self.image_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        anns = self.anns_by_image.get(image_id, [])
        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.category_id_to_label[ann["category_id"]])
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            areas_t = torch.tensor(areas, dtype=torch.float32)
            iscrowd_t = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas_t = torch.zeros((0,), dtype=torch.float32)
            iscrowd_t = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": areas_t,
            "iscrowd": iscrowd_t,
        }

        return self.to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int) -> torch.nn.Module:
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "meta": meta,
        },
        path,
    )


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def sanitize_args(args: argparse.Namespace) -> dict:
    safe = {}
    for key, value in vars(args).items():
        if callable(value):
            continue
        safe[key] = value
    return safe


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_iou50(model, loader, device, score_thresh: float = 0.3):
    model.eval()
    tp = 0
    fp = 0
    fn = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]

            keep = output["scores"].cpu() >= score_thresh
            pred_boxes = output["boxes"].cpu()[keep]
            pred_labels = output["labels"].cpu()[keep]

            if len(gt_boxes) == 0:
                fp += len(pred_boxes)
                continue

            if len(pred_boxes) == 0:
                fn += len(gt_boxes)
                continue

            ious = box_iou(pred_boxes, gt_boxes)
            matched_gt = set()

            for p_idx in range(len(pred_boxes)):
                best_iou = 0.0
                best_g = -1
                for g_idx in range(len(gt_boxes)):
                    if g_idx in matched_gt:
                        continue
                    if pred_labels[p_idx].item() != gt_labels[g_idx].item():
                        continue
                    iou_val = float(ious[p_idx, g_idx].item())
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_g = g_idx

                if best_iou >= 0.5 and best_g >= 0:
                    tp += 1
                    matched_gt.add(best_g)
                else:
                    fp += 1

            fn += (len(gt_boxes) - len(matched_gt))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def format_prediction_records(output, label_to_category_id, category_id_to_name, score_thresh=0.3):
    scores = output["scores"].cpu()
    boxes = output["boxes"].cpu()
    labels = output["labels"].cpu()

    keep = scores >= score_thresh
    records = []
    for box, label, score in zip(boxes[keep], labels[keep], scores[keep]):
        score_v = float(score.item())
        cat_id = label_to_category_id[int(label.item())]
        cat_name = category_id_to_name[cat_id]
        records.append(
            {
                "label": int(label.item()),
                "category_id": int(cat_id),
                "category_name": cat_name,
                "score": score_v,
                # Confidence-derived uncertainty proxy.
                "uncertainty": float(1.0 - score_v),
                "bbox_xyxy": [float(v) for v in box.tolist()],
            }
        )
    return records


def draw_predictions(image_path: Path, records: List[dict], output_path: Path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for rec in records:
        x1, y1, x2, y2 = rec["bbox_xyxy"]
        label = rec["category_name"]
        score = rec["score"]
        unc = rec["uncertainty"]
        color = (255, 0, 0) if unc >= 0.5 else (0, 255, 0)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{label} s={score:.2f} u={unc:.2f}"
        draw.text((x1 + 2, max(0, y1 - 12)), text, fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def cmd_inspect(args):
    ann = Path(args.ann_file)
    data = json.loads(ann.read_text(encoding="utf-8"))

    print(f"ann_file: {ann}")
    print(f"images: {len(data['images'])}")
    print(f"annotations: {len(data['annotations'])}")
    print(f"categories: {len(data['categories'])}")
    print("category names:")
    for c in data["categories"]:
        print(f"  - {c['id']}: {c['name']}")


def cmd_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")

    train_ds = VinBigCXRCOCODataset(Path(args.train_ann), Path(args.train_images))
    val_ds = VinBigCXRCOCODataset(
        Path(args.val_ann),
        Path(args.val_images),
        category_id_to_label=train_ds.category_id_to_label,
    )

    if args.max_train_samples > 0:
        train_ids = list(range(min(args.max_train_samples, len(train_ds))))
        train_ds = torch.utils.data.Subset(train_ds, train_ids)

    if args.max_val_samples > 0:
        val_ids = list(range(min(args.max_val_samples, len(val_ds))))
        val_ds = torch.utils.data.Subset(val_ds, val_ids)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    if isinstance(train_ds, torch.utils.data.Subset):
        base_train_ds = train_ds.dataset
    else:
        base_train_ds = train_ds

    num_classes = len(base_train_ds.category_id_to_label) + 1
    model = build_model(num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate_iou50(model, val_loader, device, score_thresh=args.score_thresh)

        print(
            f"epoch={epoch} loss={train_loss:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
        )

        meta = {
            "category_id_to_label": base_train_ds.category_id_to_label,
            "label_to_category_id": base_train_ds.label_to_category_id,
            "category_id_to_name": base_train_ds.category_id_to_name,
            "args": sanitize_args(args),
        }
        last_path = Path(args.output_dir) / "checkpoint_last.pt"
        save_checkpoint(last_path, model, optimizer, epoch, meta)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_path = Path(args.output_dir) / "checkpoint_best.pt"
            save_checkpoint(best_path, model, optimizer, epoch, meta)


def cmd_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")

    ds = VinBigCXRCOCODataset(Path(args.ann_file), Path(args.image_dir))
    if args.max_samples > 0:
        ds = torch.utils.data.Subset(ds, list(range(min(args.max_samples, len(ds)))))

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    if isinstance(ds, torch.utils.data.Subset):
        base_ds = ds.dataset
    else:
        base_ds = ds

    model = build_model(len(base_ds.category_id_to_label) + 1)
    ckpt = load_checkpoint(Path(args.checkpoint), model)
    model.to(device)

    metrics = evaluate_iou50(model, loader, device, score_thresh=args.score_thresh)
    print(json.dumps(metrics, indent=2))

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def cmd_infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]

    category_id_to_label = {int(k): int(v) for k, v in meta["category_id_to_label"].items()} if any(
        isinstance(k, str) for k in meta["category_id_to_label"].keys()
    ) else meta["category_id_to_label"]
    label_to_category_id = {int(k): int(v) for k, v in meta["label_to_category_id"].items()} if any(
        isinstance(k, str) for k in meta["label_to_category_id"].keys()
    ) else meta["label_to_category_id"]
    category_id_to_name = {int(k): v for k, v in meta["category_id_to_name"].items()} if any(
        isinstance(k, str) for k in meta["category_id_to_name"].keys()
    ) else meta["category_id_to_name"]

    model = build_model(len(category_id_to_label) + 1)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    img_t = transforms.ToTensor()(image).to(device)

    with torch.no_grad():
        output = model([img_t])[0]

    records = format_prediction_records(
        output,
        label_to_category_id=label_to_category_id,
        category_id_to_name=category_id_to_name,
        score_thresh=args.score_thresh,
    )

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"saved predictions: {out_json}")

    if args.output_image:
        out_img = Path(args.output_image)
        draw_predictions(image_path, records, out_img)
        print(f"saved visualization: {out_img}")


def build_parser():
    parser = argparse.ArgumentParser(description="VinBigData CXR COCO training/eval/inference baseline.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect", help="Inspect COCO annotation summary")
    p_inspect.add_argument("--ann-file", required=True)
    p_inspect.set_defaults(func=cmd_inspect)

    p_train = sub.add_parser("train", help="Train detection model")
    p_train.add_argument("--train-ann", required=True)
    p_train.add_argument("--val-ann", required=True)
    p_train.add_argument("--train-images", required=True)
    p_train.add_argument("--val-images", required=True)
    p_train.add_argument("--output-dir", default="runs/vinbig")
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--batch-size", type=int, default=2)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--num-workers", type=int, default=2)
    p_train.add_argument("--score-thresh", type=float, default=0.3)
    p_train.add_argument("--max-train-samples", type=int, default=0, help="0 means full dataset")
    p_train.add_argument("--max-val-samples", type=int, default=0, help="0 means full dataset")
    p_train.add_argument("--cpu", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate checkpoint on COCO-like val set")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--ann-file", required=True)
    p_eval.add_argument("--image-dir", required=True)
    p_eval.add_argument("--output-json", default="runs/vinbig/eval_metrics.json")
    p_eval.add_argument("--score-thresh", type=float, default=0.3)
    p_eval.add_argument("--max-samples", type=int, default=0)
    p_eval.add_argument("--num-workers", type=int, default=2)
    p_eval.add_argument("--cpu", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_infer = sub.add_parser("infer", help="Run inference + uncertainty proxy on one image")
    p_infer.add_argument("--checkpoint", required=True)
    p_infer.add_argument("--image", required=True)
    p_infer.add_argument("--output-json", default="runs/vinbig/infer.json")
    p_infer.add_argument("--output-image", default="")
    p_infer.add_argument("--score-thresh", type=float, default=0.3)
    p_infer.add_argument("--cpu", action="store_true")
    p_infer.set_defaults(func=cmd_infer)

    return parser

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("cmd", choices=["inspect", "train", "eval", "infer"])
args = parser.parse_args()


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()