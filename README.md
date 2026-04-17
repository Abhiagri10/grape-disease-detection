# 🍇 Grape Disease Detection

A transfer-learning classifier for grape leaf diseases, trained on field images captured under real outdoor conditions — not controlled lab settings. Nine disease categories, one lightweight model, runs entirely on Google Colab's free T4 GPU.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abhiagri10/grape-disease-detection/blob/main/notebooks/grape_disease_classification.ipynb)
&nbsp;&nbsp;
[![Dataset](https://img.shields.io/badge/Dataset-Mendeley%20Data-blue)](https://data.mendeley.com/datasets/bsr2vzhrzr/2)
&nbsp;&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
&nbsp;&nbsp;
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
&nbsp;&nbsp;
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)](https://pytorch.org)

---

## What this does

I built this to classify grape leaf images into nine disease or health categories using MobileNetV3-Large with ImageNet pretrained weights. The backbone is mostly frozen — only the last three feature blocks and the classifier head are trained. That keeps training fast without giving up accuracy.

The dataset came from actual vineyards in Maharashtra, India, shot on a smartphone under varying light and weather. That is what makes it interesting — models trained on clean lab datasets often fall apart in the field. This one does not, at least for this crop.

**Results on the held-out test set:**

| Metric | Value |
|--------|-------|
| Test Accuracy | **98.48%** |
| Best Validation Accuracy | **98.79%** (epoch 15) |
| Training stopped at | Epoch 23 (early stopping) |
| Training time | ~60 s/epoch on T4 |
| PyTorch checkpoint size | 41.0 MB |
| ONNX export size | 0.3 MB |

---

## Disease classes

| # | Class | Test images |
|---|-------|-------------|
| 0 | Anthracnose | 17 |
| 1 | *Guignardia bidwellii* | 11 |
| 2 | Healthy | 104 |
| 3 | Insect eating | 119 |
| 4 | Mineral deficiency | 3 |
| 5 | *Plasmopara viticola* | 30 |
| 6 | *P. viticola* + *G. bidwellii* | 10 |
| 7 | *P. viticola* + insect eating | 28 |
| 8 | Powdery mildew + insect eating | 8 |

Per-class F1 ranged from 0.931 (*Plasmopara viticola*) to 1.000 (anthracnose, healthy, *Guignardia bidwellii*). The model struggled most with *P. viticola* alone — its symptoms overlap with the co-infection classes, which makes sense visually.

---

## Dataset

2,197 grape leaf images across 9 classes, split 70/15/15 (stratified). The full dataset has 9,469 images across four crops — I used only the grape subset.

**Download:** [Mendeley Data · DOI 10.17632/bsr2vzhrzr/2](https://data.mendeley.com/datasets/bsr2vzhrzr/2)

> ⚠️ The dataset is **not included** in this repo. Download it separately and place the zip in Google Drive before running the notebook.

**Credit:** Ranveersinh R. Patil Bhosale & Prof. Trishna Ugale, COEP Technological University, Pune. Published at ICECER 2025 — DOI: [10.1109/ICECER65523.2025.11400878](https://doi.org/10.1109/ICECER65523.2025.11400878).

---

## Repository layout

```
grape-disease-detection/
├── notebooks/
│   └── grape_disease_classification.ipynb   ← run this
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── configs/
│   └── config.yaml
├── data/
│   └── README.md
├── outputs/
│   └── plots/
├── requirements.txt
├── LICENSE
└── .gitignore
```

The notebook is fully self-contained and does not import from `src/`. Those modules are there for reference and standalone use.

---

## Running it

### Google Colab (recommended)

**Step 1** — Upload the dataset zip to Google Drive:
```
MyDrive/
└── PlantDiseaseDataset/
    └── Custom Dataset for Plant Species and Disease Detec/
        └── Ranveer_Realistic_dataset_2000_1126.zip
```

**Step 2** — Open the notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abhiagri10/grape-disease-detection/blob/main/notebooks/grape_disease_classification.ipynb)

**Step 3** — Runtime → Change runtime type → **T4 GPU**, then run all cells in order.

The notebook mounts Drive, extracts grape folders only (~2,197 files), trains, evaluates, and saves everything back to Drive automatically.

---

### Local setup

```bash
git clone https://github.com/Abhiagri10/grape-disease-detection.git
cd grape-disease-detection

python3 -m venv venv && source venv/bin/activate

# Install PyTorch — check pytorch.org for your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Edit `configs/config.yaml` to update the local paths, then run the notebook cells.

---

## Dependencies

```
torch >= 2.1.0          torchvision >= 0.16.0
scikit-learn >= 1.3.0   Pillow >= 10.0.0
pyyaml >= 6.0.1         matplotlib >= 3.7.0
seaborn >= 0.13.0       onnx >= 1.15.0
onnxruntime >= 1.17.0   tqdm >= 4.66.0
numpy >= 1.24.0
```

---

## Model details

**Architecture:** MobileNetV3-Large (torchvision, ImageNet1K_V1 weights)  
**Trainable params:** 2,991,849 / 4,213,561 total (71%)

**Training setup:**

| Setting | Value |
|---------|-------|
| Loss | CrossEntropyLoss (label smoothing 0.1) |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=40) |
| Precision | Mixed (torch.amp) |
| Early stopping | Patience = 8 |
| Augmentation | Random crop, flips, ±30° rotation, colour jitter, random erasing |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| ZIP not found | Check the Drive path in Cell 6 matches where you uploaded it |
| Missing class folders | Verify `NESTED_PARENT` in Cell 2 matches the zip's top-level folder |
| CUDA out of memory | Reduce `BATCH_SIZE` in Cell 2 from 32 → 16 |
| Drive mount fails | Re-run Cell 4, grant Drive access in the popup |
| No GPU / slow | Runtime → Change runtime type → T4 GPU |

---

## License

Code: MIT — see [LICENSE](LICENSE).  
Dataset: CC BY 4.0 — Ranveersinh R. Patil Bhosale & Prof. Trishna Ugale.

---

## Citation

```bibtex
@inproceedings{bhosale2025grape,
  author    = {Ranveersinh R. Patil Bhosale and Trishna Ugale},
  title     = {Plant Species and Disease Detection using Deep Learning
               on Plant Images Captured in Realistic Natural Field Conditions},
  booktitle = {International Conference on Electrical and Computer
               Engineering Researches (ICECER 2025)},
  year      = {2025},
  doi       = {10.1109/ICECER65523.2025.11400878}
}
```

---

*Built by Abhishek B. Shirodkar · [AIT Thailand](https://ait.ac.th) · [GitHub](https://github.com/Abhiagri10)*
