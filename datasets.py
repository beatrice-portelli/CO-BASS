import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image

from utils import DISEASE_COLS, SYMPTOM_COLS


# Multi-Label Dataset Class
class SeabassDataset(Dataset):
    def __init__(self, df, img_dir, target_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_cols = target_cols
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        # Read image
        img_name = self.df.loc[idx, "photo_id"]
        img_path = os.path.join(self.img_dir, img_name+".jpg")
        
        # Load via OpenCV and convert BGR -> RGB
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract multi-label targets
        labels = self.df.loc[idx, self.target_cols].values.astype(np.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(labels)


# Multi-Label Dataset Class
class OpenClipSeabassDataset(Dataset):
    def __init__(self, df, img_dir, target_cols, preprocess=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.preprocess = preprocess
        self.target_cols = target_cols
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        # Read image
        img_name = self.df.loc[idx, "photo_id"]
        img_path = os.path.join(self.img_dir, img_name+".jpg")
        
        image = Image.open(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Extract multi-label targets
        labels = self.df.loc[idx, self.target_cols].values.astype(np.float32)
        
        if self.preprocess:
            image = self.preprocess(image).unsqueeze(0)
            
        return image, torch.tensor(labels)

# Multi-Label Dataset Class
class MTLSeabassDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.disease_cols = DISEASE_COLS
        self.symptom_cols = SYMPTOM_COLS
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        # Read image
        img_name = self.df.loc[idx, "photo_id"]
        img_path = os.path.join(self.img_dir, img_name+".jpg")
        
        # Load via OpenCV and convert BGR -> RGB
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract multi-label targets
        symptom_labels = self.df.loc[idx, self.symptom_cols].values.astype(np.float32)
        disease_labels = self.df.loc[idx, self.disease_cols].values.astype(np.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(disease_labels), torch.tensor(symptom_labels)