# dataloader/dataset.py

import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset

class SignDataset(Dataset):
    def __init__(self, csv_path, encoder_path):

        df = pd.read_csv(csv_path, dtype={"label": str}, low_memory=False)
        df["label"] = df["label"].astype(str)

        with open(encoder_path, "rb") as f:
            self.encoder = pickle.load(f)

        self.labels = self.encoder.transform(df["label"].values)

        feature_cols = [c for c in df.columns if c != "label"]
        features = df[feature_cols].values.astype(np.float32)

        
        assert features.shape[1] == 30 * 126, f"Expected 3780 features, got {features.shape[1]}"
        self.data = features.reshape(-1, 30, 126)
        
        assert len(self.data) == len(self.labels), "Data/label size mismatch!"

        print(f"✅ Dataset loaded  : {len(self.data)} samples")
        print(f"   Shape per sample: {self.data[0].shape}")
        print(f"   Classes         : {list(self.encoder.classes_)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y