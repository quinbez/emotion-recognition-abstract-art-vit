import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['emotion'])
    return df, le

def get_splits(df, subset_size=None, random_state=42):
    if subset_size:
        df, _ = train_test_split(df, train_size=subset_size,
                                 random_state=random_state,
                                 stratify=df['label'])
    train_df, temp_df = train_test_split(df, test_size=0.2,
                                         random_state=random_state,
                                         stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5,
                                       random_state=random_state,
                                       stratify=temp_df['label'])
    return train_df, val_df, test_df

def get_class_weights(train_df, device):
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['label']),
        y=train_df['label']
    )
    return torch.tensor(class_weights, dtype=torch.float32).to(device)