# Dataset Placement Guide

The dataset is **not included** in this repository (5.05 GB, CC BY 4.0 license requires attribution, not redistribution restrictions).

## Download

1. Go to: https://data.mendeley.com/datasets/bsr2vzhrzr/2
2. Click **Download All (5.05 GB)**
3. Save `Ranveer_Realistic_dataset_2000_1126.zip`

## Google Drive Setup (for Colab)

Upload the zip to your Google Drive in this exact structure:

```
MyDrive/
└── PlantDiseaseDataset/
    └── Ranveer_Realistic_dataset_2000_1126.zip
```

The notebook will:
- Mount your Drive
- Selectively extract only `grape_*` folders (~2197 images)
- Save outputs back to `MyDrive/PlantDiseaseDataset/outputs/`

## Local Setup (optional)

If running locally instead of Colab:

```bash
# 1. Extract grape folders only
python -c "
from src.utils import extract_grape_only
extract_grape_only(
    zip_path='path/to/Ranveer_Realistic_dataset_2000_1126.zip',
    extract_dir='./data/grape_only',
    species_prefix='grape_',
)
"
# 2. Update configs/config.yaml:
#    dataset.extract_path: './data/grape_only'
#    outputs.*: './outputs/...'
```

## Dataset Statistics (Grape Only)

| Class | Folder Name | Images |
|-------|-------------|--------|
| Anthracnose | `grape_anthracnose` | — |
| Guignardia bidwellii | `grape_guignardia_bidwellii` | — |
| Healthy | `grape_healthy` | — |
| Insect Eating | `grape_insect_eating` | — |
| Mineral Deficiency | `grape_mineral_dificiency` | — |
| Plasmopara viticola | `grape_plasmopara_viticola` | — |
| P. viticola + G. bidwellii | `grape_plasmopara_viticola_guignardia_bidwellii` | — |
| P. viticola + Insect | `grape_plasmopara_viticola_insect_eating` | — |
| Powdery Mildew + Insect | `grape_powdery_mildew_insect_eating` | — |
| **TOTAL** | | **~2197** |

> Note: Class folder names contain a deliberate typo — `dificiency` not `deficiency`. This matches the original dataset exactly. Do not rename.
