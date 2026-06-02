# Emotion Recognition from Abstract Art Using Vision Transformers
COMP6001 - Computer Vision and Multimodal Machine Learning | Adelaide University

## Project Overview
This project implements and evaluates a Vision Transformer (ViT) for emotion recognition from abstract art using the ArtEmis dataset. Four models are compared under identical experimental conditions:

1. **ViT (from scratch):** custom implementation following Dosovitskiy et al. (2021)
2. **ResNet-50:** pretrained transfer learning baseline
3. **SVM:** Radial Basis Function (RBF) kernel on ResNet-50 CNN features
4. **Naive Bayes:** GaussianNB on HOG + HSV hand-crafted features

---

## Repository Structure
```
emotion-recognition-abstract-art-vit/
│
├── data/                        # Dataset (not tracked by git)
│   ├── images/                  # 49,056 WikiArt painting images
│   └── dataset_final.csv        # ArtEmis annotations
│
├── notebooks/
│   ├── 01_dataset_eda.ipynb     # Dataset exploration & analysis
│   ├── 02_preprocessing.ipynb  # Image preprocessing & data loading
│   ├── 03_vit_model.ipynb       # ViT implementation & training
│   ├── 04_benchmarks.ipynb      # Benchmark models
│   └── 05_results_analysis.ipynb # Results & statistical analysis
│
├── src/
│   ├── dataset.py               # ArtEmisDataset, transforms, dataloaders
│   ├── preprocessing.py         # Data loading, splits, class weights
│   ├── model_vit.py             # ViT architecture from scratch
│   ├── model_benchmarks.py      # ResNet-50, SVM, Naive Bayes
│   ├── train.py                 # Training loop, optimizer, scheduler
│   └── evaluate.py              # Evaluation loop
│
├── models/                      # Saved model weights (not tracked by git)
├── results/                     # Plots and visualisations
├── docs/
│   └── literature_review.md     # Literature review
├── requirements.txt
└── README.md
```
---
## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/quinbez/emotion-recognition-abstract-art-vit.git
cd emotion-recognition-abstract-art-vit
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Install Kaggle CLI and authenticate:
```bash
pip install kaggle
mkdir -p ~/.kaggle
# Place your kaggle.json or access_token in ~/.kaggle/
chmod 600 ~/.kaggle/access_token
```

Download the ArtEmis dataset:
```bash
kaggle datasets download -d rollas/artemis-dataset-including-10k-images -p data/ --unzip
```

### 5. Create required directories
```bash
mkdir -p models results
```

### 6. Verify MPS (Apple Silicon) or CUDA is available
```python
import torch
print(torch.backends.mps.is_available())  # Mac M-series
print(torch.cuda.is_available())           # NVIDIA GPU
```

---

## Running the Notebooks
Run notebooks in order:

```bash
# Start Jupyter
jupyter notebook

# Then open in order:
# 1. notebooks/01_dataset_eda.ipynb
# 2. notebooks/02_preprocessing.ipynb
# 3. notebooks/03_vit_model.ipynb
# 4. notebooks/04_benchmarks.ipynb
# 5. notebooks/05_results_analysis.ipynb
```

### Running on Google Colab (recommended for training)
For full training on 15,000 samples, use Google Colab with T4 GPU:

```python
# In Colab
!git clone https://github.com/quinbez/emotion-recognition-abstract-art-vit.git
%cd emotion-recognition-abstract-art-vit
!pip install -q scikit-image einops kaggle
!mkdir -p ~/.kaggle
!echo 'YOUR_KAGGLE_TOKEN' > ~/.kaggle/access_token
# Replace YOUR_KAGGLE_TOKEN with your token from kaggle.com/settings

!chmod 600 ~/.kaggle/access_token
!kaggle datasets download -d rollas/artemis-dataset-including-10k-images -p data/ --unzip
!mkdir -p models results
```

Change device in notebooks:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Change subset size for full training:
```python
train_df, val_df, test_df = get_splits(df, subset_size=15000)
```

---

## Model Architecture

### Vision Transformer (ViT)
Custom implementation following Dosovitskiy et al. (2021):

| Component | Description |
|-----------|-------------|
| Input | 224×224 RGB image |
| Patch size | 16×16 px → 196 patches |
| Embedding dim | 256 |
| Transformer depth | 4 blocks |
| Attention heads | 8 |
| MLP ratio | 2 |
| Parameters | ~2.4M |
| Output | 8 emotion classes |

---

## Computational Complexity Analysis

### ViT (from scratch)
- **Self-attention**: O(n²·d) where n=196 patches (224×224 image, 16×16 patch size), d=256 embedding dim
- **Memory**: O(n²) per attention head, scales quadratically with number of patches
- **Parameters**: ~2.4M (embed_dim=256, depth=4, num_heads=8)

### ResNet-50
- **Convolutions**: O(k²·c·h·w) per layer where k=kernel size, c=channels, h,w=feature map dimensions
- **Memory**: Fixed 23.5M parameters regardless of input resolution
- **Parameters**: 23,524,424 total, all trainable (fine-tuned)

### SVM (RBF kernel on CNN features)
- **Feature extraction**: O(n·d) where n=samples, d=2048 ResNet-50 features
- **Training**: O(n²) to O(n³) with n training samples, becomes expensive beyond 15k samples
- **Inference**: O(n·sv) where sv=number of support vectors
- **Memory**: Scales with number of support vectors, not fixed parameters

### Naive Bayes (HOG + HSV)
- **Feature extraction**: O(n·p) where p=HOG+HSV feature dimensions per image
- **Training**: O(n·d), linear with samples and feature dimensions
- **Inference**: O(d), fastest inference among all four models
- **Memory**: O(c·d) where c=8 classes, d=feature dimensions

---

## Known Limitations

1. **ViT from scratch on small data**: ViT requires large-scale pretraining to outperform CNNs. Training from scratch on 15k samples leads to slow convergence and noisy validation accuracy due to the absence of CNN-style inductive biases (locality, translation equivariance).

2. **Class imbalance**: The dataset is severely imbalanced; sadness (37.5%) vs amusement (1.8%), a 21x difference. Class-weighted loss partially mitigates this but minority classes remain underrepresented in predictions.

3. **ResNet-50 overfitting**: Fine-tuning all 23.5M parameters on 15k images leads to significant overfitting, train accuracy reaches 50%+ while val loss diverges after epoch 3. Freezing early layers or adding dropout would help.

4. **SVM scalability**: RBF-SVM with 2048-dim features trains in O(n²) to O(n³) time. Scaling beyond 15k samples would require approximate kernel methods (e.g. Nyström approximation).

5. **Truncated images**: Some WikiArt images are corrupted or truncated. Handled via `ImageFile.LOAD_TRUNCATED_IMAGES = True` but may introduce noise into training data.

6. **Subset size**: All experiments use a stratified 15k subset of the full 49k dataset. Results may differ on the full dataset, particularly for minority emotion classes.

7. **Colab session limits**: Full training requires Google Colab T4 GPU. Free tier sessions can expire mid-training, requiring results to be saved to Google Drive after each model.

---

## Evaluation Metrics
- **Primary**: F1-macro (handles class imbalance)
- **Secondary**: Per-class AUC-ROC, confusion matrix
- **Statistical test**: McNemar's test (ViT vs ResNet-50)

---

## References
See `docs/literature_review.md` for full references.