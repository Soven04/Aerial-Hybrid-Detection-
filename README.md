AERIALDET-HYBRID
Sparse RCNN + Deformable Transformer
Complete Project Reference · Version 1.0.0
Repository	Language	Framework	Status
Soven04/Aerial-Hybrid-Detection-	Python + JavaScript	FastAPI + React	Active


1. What Is This Project?
AerialDet-Hybrid is an aerial object detection system that can identify and locate objects in satellite and drone images. It uses two powerful deep learning techniques combined together — Sparse RCNN and Deformable Transformers — to detect objects like planes, ships, vehicles, bridges, and more in overhead imagery.

This project is built as a full-stack application with a Python backend running the AI model and a React frontend providing a visual dashboard where users can upload images and see detection results in real time.

USE CASE	Detect 15 types of objects in aerial/satellite images: planes, ships, vehicles, bridges, harbors, and more — using a hybrid deep learning model.

2. Purpose of the Project
Detecting objects in aerial images is much harder than detecting objects in regular photos. Here is why:

•	Objects are tiny — a car in a satellite image might be just 5×10 pixels
•	There is extreme scale variation — a plane and a car look completely different from 500m vs 50m altitude
•	Objects are densely packed — hundreds of vehicles in a parking lot all overlap
•	No standard orientation — objects face every direction, not just left/right like in normal photos

Traditional detectors like YOLO or Faster RCNN struggle with these challenges. This project solves them by combining two modern architectures that complement each other perfectly.

3. Complete Tool Stack
3.1 Backend (AI + API)

Tool	Version	Purpose
Python	3.13+	Main programming language for AI/backend
PyTorch	2.0+	Deep learning framework — runs the neural network
Torchvision	0.15+	Computer vision utilities — NMS, transforms
FastAPI	0.110+	Web framework — serves the AI as a REST API
Uvicorn	0.29+	ASGI server — runs the FastAPI application
Pillow (PIL)	10.0+	Image loading, resizing, drawing bounding boxes
NumPy	1.24+	Numerical operations for post-processing
Pydantic	2.0+	Data validation for API request/response schemas
python-multipart	0.0.9+	Handles image file uploads in the API

3.2 Frontend (UI)

Tool	Version	Purpose
React	18.3	JavaScript UI framework — builds the dashboard
Vite	5.3	Build tool — fast development server with hot reload
Lucide React	0.400	Icon library for the interface
CSS Modules	Built-in	Scoped styling for each component
Nginx	Alpine	Serves frontend in Docker production mode

3.3 DevOps & Deployment

Tool	Purpose
Git + GitHub	Version control and code hosting
GitHub Actions	CI/CD — auto-runs tests and builds on every push
Docker	Containerizes the app for consistent deployment
Docker Compose	Runs backend + frontend together with one command
VS Code	Development environment with multi-root workspace

4. Models Used — Full Explanation
4.1 The Backbone — Feature Extractor
Think of the backbone like your eyes. It looks at the raw image and extracts meaningful visual features — edges, textures, shapes, patterns.

We use a lightweight ResNet-style backbone that produces three feature maps at different scales:

•	C3 — stride 8 (detailed features, 1/8 of original image size)
•	C4 — stride 16 (medium-level features)
•	C5 — stride 32 (high-level semantic features)

WHY	Different scales capture different object sizes. Small objects appear in C3 (high resolution), large objects appear in C5 (more semantic).

4.2 FPN — Feature Pyramid Network
The backbone gives us C3, C4, C5. But these features are disconnected — C5 knows what something is, C3 knows where it is. FPN merges them by passing information from C5 back down to C3.

FPN produces P3, P4, P5, P6 — four output feature maps all with 256 channels, each enriched with both high-level and low-level information.

•	P3 → detects small objects (cars, people)
•	P4 → detects medium objects (trucks, boats)
•	P5 → detects large objects (planes, ships)
•	P6 → detects very large structures (runways, bridges)

4.3 Deformable Transformer Encoder — The Global Brain
After FPN, we flatten all four feature maps into a single sequence of tokens and feed them through a 6-layer Deformable Transformer Encoder. This is the most important innovation in the architecture.

Normal transformers attend to every token — which is too slow for the large feature maps from aerial images. Deformable attention instead learns to focus on only a few key locations per query. This makes it fast while still capturing long-range relationships.

EXAMPLE	To recognize a harbor, the model must simultaneously see the water, the docking structures, and the ships nearby. Deformable attention lets it look at all these distant regions at once.

Each encoder layer has two components:

•	Deformable Self-Attention — each feature token attends to a few learned offset positions across all feature levels
•	Feed-Forward Network (FFN) — a 2-layer MLP with GELU activation that processes each token independently

4.4 Sparse RCNN — The Detection Head
This is the core detection mechanism. Sparse RCNN replaces the traditional dense anchor approach (which generates 100,000+ candidate boxes) with just 100 learnable proposal boxes and 100 learnable proposal features.

These proposals are like the model's attention. They learn through training to position themselves where objects typically appear in aerial images.

How One Stage Works
Each of the 6 stages does the following:

1.	Self-Attention among proposals — the 100 proposals communicate with each other, sharing context
2.	RoI Align — extract a 7x7 feature patch from the encoded feature map at each proposal location
3.	Dynamic Conv — each proposal generates its own convolution kernel to process its own RoI patch
4.	FFN — further processes the updated feature
5.	Predict — outputs class scores (15 classes) and box refinements (dx, dy, dw, dh)
6.	Refine — box = box + 0.1 * tanh(delta) — boxes improve iteratively across stages

KEY INSIGHT	Dynamic Conv means each proposal uses a different convolution kernel generated from its own features. This lets one proposal specialize for planes while another specializes for ships — all in the same forward pass.

4.5 Loss Function
During training, all 6 stages are supervised simultaneously. The loss has three components:

Loss	Weight	What It Penalizes
Binary Cross-Entropy (Classification)	2.0	Wrong class predictions
L1 Loss (Box Regression)	5.0	Inaccurate box coordinates
GIoU Loss (Box Quality)	2.0	Poor overlap between predicted and ground truth box

5. Architecture — Step by Step

Step	Component	Input	Output	Purpose
1	LightBackbone	(B, 3, 640, 640)	C3, C4, C5	Extract visual features at 3 scales
2	FPN Neck	C3, C4, C5	P3, P4, P5, P6	Merge scales, normalize to 256 channels
3	Pos Encoding	P3–P6	P3–P6 + position	Tell transformer where each token is located
4	Flatten	P3–P6	(B, N_tokens, 256)	Combine all levels into one sequence
5	Transformer Enc	(B, N, 256)	(B, N, 256)	Global context via deformable attention x6
6	Reconstruct P3	(B, N, 256)	(B, 256, H3, W3)	Reshape encoded tokens back to feature map
7	Proposals	Learnable weights	(B, 100, 4) + (B, 100, 256)	100 initial box guesses + features
8	RCNN Stage 1-6	Feature map + proposals	(B, 100, 15) + (B, 100, 4)	Iterative refinement of boxes and classes
9	Inference	Last stage output	Filtered detections	Score threshold + NMS filtering

6. Project File Structure
6.1 Complete File Tree

aerialdet_full/
├── backend/
│   ├── app.py                  ← FastAPI server (REST API endpoints)
│   ├── train.py                ← Training loop with AMP + cosine LR
│   ├── inference.py            ← Inference helper + mAP evaluation
│   ├── requirements.txt        ← Python dependencies
│   ├── Dockerfile              ← Container for backend
│   └── models/
│       └── hybrid_model.py     ← FULL MODEL (760 lines)
│           ├── PositionalEncoding2D
│           ├── DeformableAttention
│           ├── TransformerEncoder
│           ├── FPN
│           ├── DynamicConv
│           ├── SparseRCNNHead
│           ├── AerialDetHybrid    ← Main model class
│           └── AerialDetLoss
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx             ← Root component, API calls
│   │   ├── App.css             ← Layout grid
│   │   ├── index.css           ← Global vars, fonts, scanlines
│   │   ├── main.jsx            ← React entry point
│   │   └── components/
│   │       ├── Header          ← Top bar with system status
│   │       ├── UploadZone      ← Drag-drop image + sliders
│   │       ├── ResultPanel     ← Annotated output image
│   │       ├── DetectionList   ← Filterable detection table
│   │       ├── ModelInfo       ← Architecture diagram tab
│   │       └── StatusBar       ← Footer with live clock
│   ├── vite.config.js          ← Dev server + API proxy
│   ├── package.json
│   ├── Dockerfile
│   └── nginx.conf              ← Production server config
│
├── .github/workflows/ci.yml   ← GitHub Actions CI/CD
├── docker-compose.yml          ← One-command full stack
├── aerialdet.code-workspace    ← VS Code workspace
└── README.md

6.2 Purpose of Each File

File	What It Does
hybrid_model.py	The entire neural network — backbone, FPN, transformer, Sparse RCNN head, loss function. This is the core of the project.
app.py	FastAPI server with /detect, /health, /classes, /model/info endpoints. Loads the model once at startup, handles image uploads, returns annotated images as base64.
train.py	Training loop with mixed precision (AMP), cosine learning rate warmup, gradient clipping, and checkpoint saving. Supports DOTA/VisDrone datasets in YOLO format.
inference.py	Standalone inference script. Also contains compute_map() for evaluating model accuracy with mAP metric.
App.jsx	Root React component. Manages all state (file, result, loading, thresholds), calls the API, and passes data to child components.
UploadZone.jsx	Handles drag-and-drop image upload. Contains the score threshold and NMS threshold sliders. Shows targeting corner brackets on the image preview.
ResultPanel.jsx	Displays the annotated output image returned by the API. Shows loading radar animation during inference. Shows detection count, inference time, and device stats.
DetectionList.jsx	Filterable table of all detected objects with class name, confidence score, color indicator, and pixel coordinates.
ModelInfo.jsx	Architecture pipeline diagram, parameter counts, class list, and key innovations — shown in the Model Info tab.
vite.config.js	Configures the Vite dev server to proxy /api/* to http://127.0.0.1:8000 so frontend can talk to backend without CORS issues.
ci.yml	GitHub Actions workflow that runs Python syntax checks and React build on every git push.

7. Code Execution Sequence
7.1 Backend Startup Sequence

7.	uvicorn starts app.py — Python imports all modules
8.	get_model() is called lazily — AerialDetHybrid(num_classes=15) is instantiated (~28M parameters)
9.	Checkpoint loading — if checkpoints/best.pth exists, weights are loaded; otherwise random weights (demo mode)
10.	FastAPI registers routes — /health, /detect, /classes, /model/info become available

7.2 Detection Request Sequence

11.	POST /detect — frontend sends image file + score_threshold + nms_threshold
12.	preprocess() — resize to 640x640, normalize with ImageNet mean/std, convert to tensor
13.	model.predict() — runs full forward pass through backbone → FPN → transformer → 6 RCNN stages
14.	postprocess() — convert normalized boxes to pixels, filter by score, apply NMS
15.	draw_boxes() — draw colored rectangles and labels on original image
16.	Return JSON — annotated image as base64, detection list, count, inference time

7.3 Model Forward Pass Sequence (inside hybrid_model.py)

images (B, 3, 640, 640)
  → LightBackbone.forward()
      → stem (conv + bn + relu + maxpool)
      → stage2, stage3, stage4, stage5
      → returns [C3, C4, C5]
  → FPN.forward([C3, C4, C5])
      → lateral_convs: 1x1 conv on each
      → top-down: upsample + add
      → output_convs: 3x3 conv on each
      → extra_conv: stride-2 conv for P6
      → returns [P3, P4, P5, P6]
  → _flatten_features([P3, P4, P5, P6])
      → PositionalEncoding2D on each level
      → flatten H*W dimension
      → concatenate all levels
      → returns flat_src (B, N_total, 256)
  → TransformerEncoder.forward()
      → 6x DeformableAttention + FFN
      → returns encoded (B, N_total, 256)
  → reconstruct P3 map (B, 256, H3, W3)
  → SparseRCNNHead.forward()
      → 6x SparseSingleStage:
          → self_attn (proposals attend each other)
          → _roi_align (7x7 grid_sample)
          → DynamicConv (instance kernels)
          → FFN
          → cls_head, reg_head
          → box refinement
      → returns list of 6 stage outputs

8. API Reference

Method	Endpoint	Parameters	Returns
GET	/health	None	{"status": "ok", "device": "cpu", "checkpoint": null}
GET	/classes	None	List of 15 DOTA classes with id and color
GET	/model/info	None	Architecture details: params, layers, config
POST	/detect	file (image), score_threshold (0-1), nms_threshold (0-1)	Annotated image (base64), detections list, count, inference_ms

8.1 Detection Response Format

{
  "image_b64": "data:image/png;base64,...",
  "detections": [
    {
      "label": "ship",
      "label_id": 6,
      "score": 0.8721,
      "box": [x1, y1, x2, y2],   // pixel coordinates
      "color": "#42d4f4"
    }
  ],
  "count": 12,
  "inference_ms": 342.5,
  "device": "cpu",
  "image_w": 1024,
  "image_h": 768
}

9. DOTA Dataset Classes
The model is designed for DOTA v1.0 — the largest aerial object detection dataset with 15 categories:

ID	Class Name	Description
0	plane	Commercial/military aircraft on runways or taxiways
1	baseball-diamond	Baseball fields visible from above
2	bridge	Road or rail bridges over water/valleys
3	ground-track-field	Athletic running tracks
4	small-vehicle	Cars, SUVs, small trucks
5	large-vehicle	Trucks, buses, large commercial vehicles
6	ship	Vessels in water — cargo, tanker, ferry
7	tennis-court	Tennis courts
8	basketball-court	Basketball courts
9	storage-tank	Cylindrical oil/chemical storage tanks
10	soccer-ball-field	Football/soccer pitches
11	roundabout	Circular road intersections
12	harbor	Docking areas, piers, ports
13	swimming-pool	Residential or public swimming pools
14	helicopter	Rotary-wing aircraft

10. Setup & Run Guide
10.1 Prerequisites
•	Python 3.11 or higher
•	Node.js 18 or higher (LTS)
•	Git
•	8GB+ RAM recommended for running the model

10.2 Local Development
Backend
cd aerialdet_full/backend
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
source .venv/bin/activate            # Mac/Linux
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

Frontend
cd aerialdet_full/frontend
npm install
npm run dev
# Open http://localhost:5173

10.3 Docker (One Command)
docker-compose up
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs

10.4 Training Your Own Model
Prepare your data in YOLO format:

data/aerial/
  images/  img001.jpg  img002.jpg ...
  labels/  img001.txt  img002.txt ...

# Each .txt line: class_id cx cy w h  (normalized 0-1)
# Example: 4 0.512 0.437 0.032 0.019

python train.py --data data/aerial --epochs 36

Best checkpoint saved to checkpoints/best.pth automatically.

11. Key Innovations Explained Simply

Innovation	Simple Explanation	Why It Matters for Aerial
Sparse Proposals	100 learned box guesses replace 100,000+ grid anchors	Massively reduces computation on large aerial tiles
Dynamic Conv	Each of the 100 proposals generates its own unique convolution kernel	One proposal specializes for planes, another for ships — better accuracy
Iterative Refinement	6 stages progressively move boxes closer to the true object location	Small objects need multiple corrections to localize precisely
Deformable Attention	Attention samples a few key positions rather than all positions	Efficient long-range context — sees entire scene to understand each object
Multi-Scale FPN	Merges features from 4 pyramid levels	Handles objects from 5px cars to 200px runways in the same image
GIoU Loss	Penalizes boxes that do not overlap well, even when far apart	Better gradient signal for tiny aerial objects than basic L1 loss

12. GitHub Repository

REPO URL	https://github.com/Soven04/Aerial-Hybrid-Detection-

BRANCH	main — all production code lives here

CI/CD	GitHub Actions runs automatically on every push to main — checks Python syntax and builds the React frontend

12.1 How to Push Updates
cd aerialdet_full
git add .
git commit -m "your message here"
git push

12.2 Git History

Commit	Message	What Changed
Initial	feat: AerialDet-Hybrid - Sparse RCNN + Transformer	Full project — model, API, frontend, CI/CD
Latest	fix: update git author	Author name corrected to Soven04

13. Troubleshooting Reference

Error	Cause	Fix
Could not import module "app"	Running uvicorn from wrong folder	cd into backend/ first, then run uvicorn
API: OFFLINE in UI	Backend not running or wrong proxy URL	Check vite.config.js — target must be http://127.0.0.1:8000
Server error 500	python-multipart missing or model crash	pip install python-multipart, then restart uvicorn
npm not recognized	Node.js not installed or not in PATH	Download from nodejs.org, restart VS Code after install
python not found	Python not in PATH	Reinstall Python with "Add to PATH" checked
git push rejected	Remote has commits not in local	git push --force to overwrite remote with local
.venv activation blocked	PowerShell execution policy	Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

14. Model Parameters Summary

Parameter	Value
Total Parameters	~28 Million
Input Size	640 × 640 pixels
Feature Dimension	256 channels
Attention Heads	8
Transformer Encoder Layers	6
Sparse RCNN Stages	6
Learnable Proposals	100
RoI Size	7 × 7
Output Classes	15 (DOTA v1.0)
Training Epochs	36 (default)
Learning Rate	1e-4 (cosine schedule)
Batch Size	4
Mixed Precision	Yes (AMP)

AerialDet-Hybrid · Soven04 · github.com/Soven04/Aerial-Hybrid-Detection-
