
# AgroPest-12 Insect Detection & Classification (PyTorch)

Two modular pipelines:
- **Pipeline A**: RT-DETR (detector) → crop → ViT-B/16 (classifier) + attention visualization.
- **Pipeline B**: RetinaNet (ResNet50-FPN + Focal Loss) end-to-end + Grad-CAM.

> Works on local GPU. Python ≥3.10 recommended.

## Quickstart

```bash
# 1) Create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Edit config
vim config.yaml  # set dataset paths etc.

# 3) Train detectors
python src/detection/train_rtdetr.py --config config.yaml
python src/detection/train_retinanet.py --config config.yaml

# 4) Prepare crops and train classifier (Pipeline A)
python src/pipelines/prepare_crops_from_gt.py --config config.yaml
python src/classifier/train_vit.py --config config.yaml

# 5) Inference
python src/pipelines/infer_pipelineA.py --config config.yaml --image examples/sample.jpg
python src/pipelines/infer_retinanet.py --config config.yaml --image examples/sample.jpg

# 6) Evaluate
python src/utils/eval_detection.py --config config.yaml --pipeline retinanet
python src/utils/eval_classification.py --config config.yaml
```

## Layout
- `src/detection/` — RT-DETR (toy) + RetinaNet training/inference
- `src/classifier/` — ViT classifier
- `src/pipelines/` — end-to-end scripts (crop + infer)
- `src/utils/` — datasets, metrics, seed, viz
- `src/xai/` — attention rollout (ViT), Grad-CAM (RetinaNet)
- `src/robust/` — image distortions

This codebase is intentionally **lightweight** for coursework: easy to read, extend, and cite.
