import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage.color import rgb2hsv
import torchvision.models as models

# ResNet-50
def get_resnet50(num_classes=8):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# SVM
def get_feature_extractor(device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    return feature_extractor.to(device)


def extract_features(loader, feature_extractor, device):
    features, labels_list = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = feature_extractor(images)
            feats = feats.squeeze(-1).squeeze(-1)
            features.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())
    return np.concatenate(features), np.concatenate(labels_list)

def train_svm(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    svm = SVC(kernel='rbf', class_weight='balanced', probability=True)
    svm.fit(X_scaled, y_train)
    return svm, scaler

# Naive Bayes
def extract_hsv_histogram(image, bins=32):
    hsv = rgb2hsv(np.array(image))
    h_hist, _ = np.histogram(hsv[:,:,0], bins=bins, range=(0,1))
    s_hist, _ = np.histogram(hsv[:,:,1], bins=bins, range=(0,1))
    v_hist, _ = np.histogram(hsv[:,:,2], bins=bins, range=(0,1))
    hist = np.concatenate([h_hist, s_hist, v_hist]).astype(float)
    return hist / hist.sum()

def extract_hog_features(image, image_size=(64, 64)):
    img_resized = image.resize(image_size)
    img_gray = np.array(img_resized.convert('L'))
    features = hog(img_gray, orientations=8,
                   pixels_per_cell=(8,8), cells_per_block=(2,2))
    return features

def extract_handcrafted_features(dataframe, image_dir):
    features, labels = [], []
    image_dir = Path(image_dir)
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        try:
            image = Image.open(image_dir / row['filename']).convert('RGB')
            hsv_feats = extract_hsv_histogram(image)
            hog_feats = extract_hog_features(image)
            combined = np.concatenate([hsv_feats, hog_feats])
            features.append(combined)
            labels.append(row['label'])
        except Exception:
            continue
    return np.array(features), np.array(labels)

def train_naive_bayes(X_train, y_train):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    return nb