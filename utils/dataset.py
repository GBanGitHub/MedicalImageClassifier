import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms

class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            split (string): 'train' or 'val' to specify the dataset split
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Load metadata
        self.metadata = pd.read_csv(os.path.join(root_dir, 'metadata.csv'))
        
        # Filter for split
        self.metadata = self.metadata[self.metadata['split'] == split]
        
        # Get image paths and labels
        self.image_paths = self.metadata['image_path'].values
        self.labels = self.metadata['label'].values
        
        # Create label to index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load image
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Get label
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx
    
    def get_label_mapping(self):
        return {idx: label for label, idx in self.label_to_idx.items()} 