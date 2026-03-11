# AerialDet-Hybrid 🛰️
**Sparse RCNN + Deformable Transformer for Aerial Object Detection**

---

## Quick Start (3 commands)

```bash
git clone https://github.com/YOUR_USERNAME/aerialdet-hybrid.git
cd aerialdet-hybrid
docker-compose up
```

Then open **http://localhost:3000**

---

## Project Structure

```
aerialdet-hybrid/
├── backend/
│   ├── app.py                   ← FastAPI server (REST API)
│   ├── train.py                 ← Training loop
│   ├── inference.py             ← Inference + mAP eval
│   ├── requirements.txt
│   ├── Dockerfile
│   └── models/
│       └── hybrid_model.py      ← Full model architecture
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx              ← Root component
│   │   ├── components/
│   │   │   ├── Header.jsx       ← Top nav + status
│   │   │   ├── UploadZone.jsx   ← Image upload + controls
│   │   │   ├── ResultPanel.jsx  ← Detection output viewer
│   │   │   ├── DetectionList.jsx← Tabular results
│   │   │   ├── ModelInfo.jsx    ← Architecture viewer
│   │   │   └── StatusBar.jsx    ← Footer stats
│   │   └── index.css
│   ├── package.json
│   ├── vite.config.js
│   ├── Dockerfile
│   └── nginx.conf
│
├── .github/workflows/ci.yml     ← GitHub Actions CI
├── docker-compose.yml
├── aerialdet.code-workspace     ← VS Code workspace
└── README.md
```

---

## Development Setup (without Docker)

### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### Frontend
```bash
cd frontend
npm install
npm run dev                        # → http://localhost:5173
```

---

## API Endpoints

| Method | Route          | Description                        |
|--------|---------------|------------------------------------|
| GET    | /health       | Server health + device info        |
| GET    | /classes      | All 15 DOTA class names + colors   |
| GET    | /model/info   | Architecture summary               |
| POST   | /detect       | Run detection on uploaded image    |

**POST /detect** params:
- `file`: image (multipart)
- `score_threshold`: float 0–1 (default 0.35)
- `nms_threshold`: float 0–1 (default 0.50)

---

## GitHub Deployment

```bash
git init
git add .
git commit -m "feat: initial AerialDet-Hybrid implementation"
git remote add origin https://github.com/YOUR_USERNAME/aerialdet-hybrid.git
git push -u origin main
```

GitHub Actions will automatically:
1. Lint + syntax-check the backend Python files
2. Build the React frontend
3. Build Docker images (on `main` branch)

---

## Training

```bash
cd backend

# Prepare data (YOLO format)
# data/aerial/images/*.jpg
# data/aerial/labels/*.txt  (class cx cy w h)

python train.py --data data/aerial --epochs 36
```

Checkpoints saved to `backend/checkpoints/best.pth`

---

## Model Architecture

```
Input → ResNet Backbone (C3/C4/C5)
      → FPN Neck (P3–P6, 256ch)
      → Deformable Transformer Encoder (6 layers)
      → 100 Learnable Proposals
      → Sparse RCNN Dynamic Head (6 iterative stages)
      → Final: cls_logits (B,100,15) + boxes (B,100,4)
```

| Metric              | Value     |
|---------------------|-----------|
| Parameters          | ~28M      |
| Proposals           | 100       |
| RCNN stages         | 6         |
| Encoder layers      | 6         |
| Feature dim         | 256       |
| DOTA classes        | 15        |
| Input size          | 640×640   |
