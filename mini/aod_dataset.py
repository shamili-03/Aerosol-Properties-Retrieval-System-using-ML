import torch
from torch.utils.data import Dataset
import pandas as pd
import rasterio
import numpy as np

class AODDataset(Dataset):
    def __init__(self, csv_path, image_dir):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = f"{self.image_dir}/{row['filename']}"
        aod_value = row['AOD']

        with rasterio.open(filename) as src:
            image = src.read()

        if image.shape[0] < 13:
            pad = 13 - image.shape[0]
            image = np.pad(image, ((0, pad), (0, 0), (0, 0)), mode='constant')

        image = image[:13]
        image = torch.tensor(image, dtype=torch.float32)
        return image, torch.tensor(aod_value, dtype=torch.float32)
