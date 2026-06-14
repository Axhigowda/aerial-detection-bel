# Object Detection using UAV Imagery: YOLO26 vs RT-DETR
### BEL (Bharat Electronics Limited) Internship Project

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/Axhigowda/aerial-detection-bel)

> 🚀 **Try the live demo:** [huggingface.co/spaces/Axhigowda/aerial-detection-bel](https://huggingface.co/spaces/Axhigowda/aerial-detection-bel)

## Overview
Comparative study of CNN-based (YOLO26s) and Transformer-based (RT-DETR-L) architectures for real-time aerial object detection on UAV imagery using the VisDrone2019 dataset.

## Results
| Model | mAP@50 | mAP@50-95 | Precision | Recall | Inference |
|-------|--------|-----------|-----------|--------|-----------|
| **YOLO26s** | **0.368** | **0.211** | **0.486** | **0.379** | **3.0ms** |
| RT-DETR-L | 0.072 | 0.035 | 0.231 | 0.182 | 14.3ms |

## Dataset
- **VisDrone2019** — 6,471 training images, 548 validation images
- Captured by drone cameras across 14 cities in China
- 10 object classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor
- **Not included in YOLO pretrained COCO classes** (aerial perspective)

## Models
- **YOLO26s** — Latest Ultralytics YOLO (Jan 2026), NMS-free, edge-optimized, 640px
- **RT-DETR-L** — Real-Time Detection Transformer with AIFI attention encoder, 320px

## Key Findings
- YOLO26s outperforms RT-DETR-L by **5x on mAP@50** on this dataset
- YOLO26s is **4.8x faster** (3.0ms vs 14.3ms per image)
- Car detection is strongest for both models (largest objects in aerial view)
- Bicycle detection is hardest (smallest objects, <32px in aerial view)
- RT-DETR was constrained to 320px due to T4 GPU memory limits

## Project Structure
```
aerial-detection-bel/
├── results/
│   ├── comparison_chart.png
│   ├── perclass_comparison.png
│   ├── yolo26_detections.png
│   ├── rtdetr_detections.png
│   ├── yolo26_metrics.json
│   └── rtdetr_metrics.json
├── .gitignore
├── LICENSE
└── README.md
```

## Tech Stack
- Python 3.12, PyTorch 2.10, Ultralytics 8.4
- YOLO26s (9.9M params, 22.5 GFLOPs)
- RT-DETR-L (32.8M params, 108 GFLOPs)
- Training: Kaggle T4 GPU

## How to Run
```bash
pip install ultralytics
```
```python
from ultralytics import YOLO
model = YOLO('weights/yolo26s_best.pt')
results = model('your_drone_image.jpg', imgsz=640)
results[0].show()
```

## Author
Internship Project — Bharat Electronics Limited (BEL), Bangalore
Domain: AI/ML | June 2026