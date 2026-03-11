"""
Inference & Evaluation for AerialDet-Hybrid
"""

import torch
import torchvision
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from torchvision import transforms
from models.hybrid_model import AerialDetHybrid


# DOTA v1.0 class names
DOTA_CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field',
    'roundabout', 'harbor', 'swimming-pool', 'helicopter'
]

COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
]


def load_model(checkpoint_path: str, cfg: dict, device: str = 'cpu'):
    model = AerialDetHybrid(
        num_classes   = cfg.get('num_classes', 15),
        num_proposals = cfg.get('num_proposals', 100),
        d_model       = cfg.get('d_model', 256),
        n_heads       = cfg.get('n_heads', 8),
        num_enc_layers= cfg.get('num_enc_layers', 6),
        num_stages    = cfg.get('num_stages', 6),
        roi_size      = cfg.get('roi_size', 7),
    )
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"[AerialDet] Loaded checkpoint: {checkpoint_path}")
    model = model.to(device)
    return model


def preprocess(image_path: str, img_size: int = 640):
    img = Image.open(image_path).convert('RGB')
    orig_size = img.size   # (W, H)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0)   # (1, 3, H, W)
    return tensor, img, orig_size


def postprocess(predictions: dict, orig_size: tuple,
                score_thresh: float = 0.4,
                nms_thresh:   float = 0.5):
    """Convert normalized cx,cy,w,h → pixel x1,y1,x2,y2 and apply NMS."""
    W, H  = orig_size
    scores = predictions['scores'].cpu()
    labels = predictions['labels'].cpu()
    boxes  = predictions['boxes'].cpu()   # cx, cy, w, h in [0,1]

    if len(boxes) == 0:
        return [], [], []

    # Convert to x1y1x2y2 pixels
    cx, cy, w, h = boxes.unbind(-1)
    x1 = ((cx - w/2) * W).clamp(0, W)
    y1 = ((cy - h/2) * H).clamp(0, H)
    x2 = ((cx + w/2) * W).clamp(0, W)
    y2 = ((cy + h/2) * H).clamp(0, H)
    xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

    # Score filter
    keep = scores > score_thresh
    xyxy, scores, labels = xyxy[keep], scores[keep], labels[keep]

    # NMS
    if len(xyxy) > 0:
        nms_keep = torchvision.ops.nms(xyxy.float(), scores.float(), nms_thresh)
        xyxy   = xyxy[nms_keep].numpy()
        scores = scores[nms_keep].numpy()
        labels = labels[nms_keep].numpy()
    else:
        xyxy, scores, labels = np.array([]), np.array([]), np.array([])

    return xyxy, scores, labels


def visualize(image: Image.Image, xyxy, scores, labels,
              class_names=DOTA_CLASSES, out_path: str = None) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for box, score, label in zip(xyxy, scores, labels):
        x1, y1, x2, y2 = box
        color = COLORS[int(label) % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        name = class_names[int(label)] if int(label) < len(class_names) else str(label)
        text = f"{name} {score:.2f}"
        draw.text((x1, max(y1 - 16, 0)), text, fill=color, font=font)

    if out_path:
        image.save(out_path)
        print(f"[AerialDet] Saved visualization: {out_path}")
    return image


def run_inference(image_path: str,
                  checkpoint_path: str = None,
                  img_size: int = 640,
                  score_thresh: float = 0.4,
                  nms_thresh: float = 0.5,
                  out_path: str = 'output_detection.jpg'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(checkpoint_path, {}, device)
    tensor, orig_img, orig_size = preprocess(image_path, img_size)
    tensor = tensor.to(device)

    predictions = model.predict(tensor, score_thresh)[0]
    xyxy, scores, labels = postprocess(predictions, orig_size,
                                        score_thresh, nms_thresh)

    print(f"[AerialDet] Detected {len(xyxy)} objects in {image_path}")
    for i, (box, s, l) in enumerate(zip(xyxy, scores, labels)):
        name = DOTA_CLASSES[int(l)] if int(l) < len(DOTA_CLASSES) else l
        print(f"  [{i+1}] {name:25s} score={s:.3f}  box={box.astype(int)}")

    result_img = visualize(orig_img.copy(), xyxy, scores, labels,
                           out_path=out_path)
    return result_img, xyxy, scores, labels


# ─────────────────────────────────────────────
#  mAP Evaluation
# ─────────────────────────────────────────────
def compute_map(gt_list, pred_list, iou_threshold=0.5, num_classes=15):
    """
    Compute mAP@IoU_threshold.
    gt_list:   [{'boxes': (M,4), 'labels': (M,)}]
    pred_list: [{'boxes': (N,4), 'scores': (N,), 'labels': (N,)}]
    Boxes in xyxy format.
    """
    aps = []
    for cls in range(num_classes):
        tp_list, fp_list, scores_list = [], [], []
        n_gt = 0

        for gt, pred in zip(gt_list, pred_list):
            gt_mask   = gt['labels'] == cls
            pred_mask = pred['labels'] == cls

            gt_boxes   = gt['boxes'][gt_mask]
            pred_boxes = pred['boxes'][pred_mask]
            pred_scores= pred['scores'][pred_mask]

            n_gt += len(gt_boxes)
            if len(pred_boxes) == 0:
                continue

            # Sort by score descending
            order = pred_scores.argsort()[::-1]
            pred_boxes  = pred_boxes[order]
            pred_scores = pred_scores[order]

            matched = np.zeros(len(gt_boxes), dtype=bool)
            for pb in pred_boxes:
                if len(gt_boxes) == 0:
                    tp_list.append(0); fp_list.append(1)
                    continue
                # Compute IoU
                ix1 = np.maximum(pb[0], gt_boxes[:, 0])
                iy1 = np.maximum(pb[1], gt_boxes[:, 1])
                ix2 = np.minimum(pb[2], gt_boxes[:, 2])
                iy2 = np.minimum(pb[3], gt_boxes[:, 3])
                inter = np.maximum(ix2-ix1, 0) * np.maximum(iy2-iy1, 0)
                area_p = (pb[2]-pb[0]) * (pb[3]-pb[1])
                area_g = (gt_boxes[:,2]-gt_boxes[:,0]) * (gt_boxes[:,3]-gt_boxes[:,1])
                iou = inter / (area_p + area_g - inter + 1e-6)

                best_idx = iou.argmax()
                if iou[best_idx] >= iou_threshold and not matched[best_idx]:
                    tp_list.append(1); fp_list.append(0)
                    matched[best_idx] = True
                else:
                    tp_list.append(0); fp_list.append(1)

            scores_list.extend(pred_scores.tolist())

        if n_gt == 0:
            continue

        tp = np.array(tp_list)
        fp = np.array(fp_list)
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall    = cum_tp / (n_gt + 1e-6)
        precision = cum_tp / (cum_tp + cum_fp + 1e-6)

        # AP via 11-point interpolation
        ap = 0.
        for thresh in np.linspace(0, 1, 11):
            mask = recall >= thresh
            ap += precision[mask].max() if mask.any() else 0.
        ap /= 11.
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [checkpoint_path]")
        print("\nRunning model summary instead...")

        # Quick model summary
        model = AerialDetHybrid(num_classes=15, num_proposals=100)
        total = sum(p.numel() for p in model.parameters()) / 1e6
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"\nAerialDet-Hybrid Architecture")
        print(f"  Total params:     {total:.1f}M")
        print(f"  Trainable params: {trainable:.1f}M")

        # Forward pass test
        dummy = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            outputs = model(dummy)
        print(f"\nForward pass ✓")
        print(f"  Stages: {len(outputs)}")
        print(f"  Final stage cls_logits: {outputs[-1]['cls_logits'].shape}")
        print(f"  Final stage boxes:      {outputs[-1]['boxes'].shape}")
    else:
        img_path = sys.argv[1]
        ckpt     = sys.argv[2] if len(sys.argv) > 2 else None
        run_inference(img_path, ckpt)
