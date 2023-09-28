import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

class CustomDataset(Dataset):
    """My CustomDataset to load the images per folder"""
    def __init__(self, root_dir, transforms = None):
        super(CustomDataset, self).__init__()
        self.root_dir = root_dir
        self.transforms = transforms
        self.class_to_idx = CustomDataset._class_to_idx(self.root_dir)
        self.image_paths = self._load_image_paths()
    
    @classmethod
    def _class_to_idx(cls, root_dir:str):
        class_to_idx = {}
        for idx, class_dir in enumerate(os.listdir(root_dir)):
            class_to_idx[class_dir] = idx
        return class_to_idx
    
    
    def _load_image_paths(self):
        image_paths = []
        for classname in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, classname)
            if os.path.isdir(class_dir):
                class_idx = self.class_to_idx[classname]
                for filename in os.listdir(class_dir):
                    image_paths.append((os.path.join(class_dir, filename), class_idx))
        return image_paths
        
    def __getitem__(self, index):
        img_path, label = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')
        
        if self.transforms:
            image = self.transforms(image)
            
        label_tensor = torch.zeros(len(self.class_to_idx))
        label_tensor[label] = 1
        
        
        return image, label_tensor

    def __len__(self):
        return len(self.image_paths)
