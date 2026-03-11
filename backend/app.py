"""
AerialDet-Hybrid  ·  FastAPI Backend
=====================================
Endpoints:
  POST /detect          — run detection on uploaded image
  GET  /health          — health check
  GET  /classes         — list DOTA class names
  GET  /model/info      — model architecture summary
"""

import io
import time
import base64
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import torchvision
from torchvision import transforms

from models.hybrid_model import AerialDetHybrid

# ─────────────────────────────────────────────
#  App & CORS
# ─────────────────────────────────────────────
app = FastAPI(
    title="AerialDet-Hybrid API",
    description="Sparse RCNN + Transformer aerial object detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
DOTA_CLASSES = [
    "plane", "baseball-diamond", "bridge", "ground-track-field",
    "small-vehicle", "large-vehicle", "ship", "tennis-court",
    "basketball-court", "storage-tank", "soccer-ball-field",
    "roundabout", "harbor", "swimming-pool", "helicopter",
]

COLORS = [
    "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231",
    "#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4",
    "#469990","#dcbeff","#9A6324","#fffac8","#800000",
]

IMG_SIZE    = 640
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT  = Path("checkpoints/best.pth")

# ─────────────────────────────────────────────
#  Model (singleton)
# ─────────────────────────────────────────────
_model: Optional[AerialDetHybrid] = None

def get_model() -> AerialDetHybrid:
    global _model
    if _model is None:
        _model = AerialDetHybrid(num_classes=15, num_proposals=100).to(DEVICE)
        if CHECKPOINT.exists():
            ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
            _model.load_state_dict(ckpt["model"])
            print(f"[API] Loaded checkpoint: {CHECKPOINT}")
        else:
            print("[API] No checkpoint found — using random weights (demo mode)")
        _model.eval()
    return _model

# ─────────────────────────────────────────────
#  Image helpers
# ─────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def preprocess(pil_img: Image.Image) -> torch.Tensor:
    return _transform(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)

def postprocess(preds: dict, orig_w: int, orig_h: int,
                score_thresh: float, nms_thresh: float):
    scores = preds["scores"].cpu()
    labels = preds["labels"].cpu()
    boxes  = preds["boxes"].cpu()

    if len(boxes) == 0:
        return [], [], []

    cx, cy, w, h = boxes.unbind(-1)
    x1 = ((cx - w / 2) * orig_w).clamp(0, orig_w)
    y1 = ((cy - h / 2) * orig_h).clamp(0, orig_h)
    x2 = ((cx + w / 2) * orig_w).clamp(0, orig_w)
    y2 = ((cy + h / 2) * orig_h).clamp(0, orig_h)
    xyxy = torch.stack([x1, y1, x2, y2], -1)

    keep = scores > score_thresh
    xyxy, scores, labels = xyxy[keep], scores[keep], labels[keep]

    if len(xyxy) > 0:
        nms_keep = torchvision.ops.nms(xyxy.float(), scores.float(), nms_thresh)
        xyxy   = xyxy[nms_keep].tolist()
        scores = scores[nms_keep].tolist()
        labels = labels[nms_keep].tolist()
    else:
        xyxy, scores, labels = [], [], []

    return xyxy, scores, labels

def draw_boxes(img: Image.Image, xyxy, scores, labels) -> str:
    """Draw boxes and return base64-encoded PNG."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for box, score, label in zip(xyxy, scores, labels):
        x1, y1, x2, y2 = box
        color = COLORS[int(label) % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        name  = DOTA_CLASSES[int(label)] if int(label) < len(DOTA_CLASSES) else str(label)
        draw.text((x1 + 4, y1 + 2), f"{name} {score:.2f}", fill=color, font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# ─────────────────────────────────────────────
#  Schemas
# ─────────────────────────────────────────────
class Detection(BaseModel):
    label: str
    label_id: int
    score: float
    box: List[float]   # [x1, y1, x2, y2] pixels
    color: str

class DetectResponse(BaseModel):
    image_b64:    str
    detections:   List[Detection]
    count:        int
    inference_ms: float
    device:       str
    image_w:      int
    image_h:      int

# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE,
            "checkpoint": str(CHECKPOINT) if CHECKPOINT.exists() else None}

@app.get("/classes")
def get_classes():
    return {"classes": [{"id": i, "name": n, "color": COLORS[i]}
                         for i, n in enumerate(DOTA_CLASSES)]}

@app.get("/model/info")
def model_info():
    model = get_model()
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "architecture": "AerialDet-Hybrid (Sparse RCNN + Deformable Transformer)",
        "total_params_M":     round(total / 1e6, 2),
        "trainable_params_M": round(trainable / 1e6, 2),
        "num_classes":    15,
        "num_proposals":  100,
        "num_stages":     6,
        "encoder_layers": 6,
        "d_model":        256,
        "device":         DEVICE,
    }

@app.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(...),
    score_threshold: float = Query(0.35, ge=0.0, le=1.0),
    nms_threshold:   float = Query(0.50, ge=0.0, le=1.0),
):
    # Read image
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    orig_w, orig_h = img.size

    # Inference
    model  = get_model()
    tensor = preprocess(img)

    t0 = time.perf_counter()
    with torch.no_grad():
        preds = model.predict(tensor, score_threshold=score_threshold)[0]
    ms = (time.perf_counter() - t0) * 1000

    # Post-process
    xyxy, scores, labels = postprocess(preds, orig_w, orig_h,
                                        score_threshold, nms_threshold)

    detections = [
        Detection(
            label=DOTA_CLASSES[int(l)] if int(l) < len(DOTA_CLASSES) else str(l),
            label_id=int(l),
            score=round(float(s), 4),
            box=[round(v, 1) for v in b],
            color=COLORS[int(l) % len(COLORS)],
        )
        for b, s, l in zip(xyxy, scores, labels)
    ]

    # Draw annotated image
    annotated_b64 = draw_boxes(img.copy(), xyxy, scores, labels)

    return DetectResponse(
        image_b64=annotated_b64,
        detections=detections,
        count=len(detections),
        inference_ms=round(ms, 1),
        device=DEVICE,
        image_w=orig_w,
        image_h=orig_h,
    )

# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
