import torch
from torch import nn
import torchvision.transforms as T

class CarClassifierV0(nn.Module):
    def __init__(self, num_classes, mean, std):
        super(CarClassifierV0, self).__init__()
        self.num_classes = num_classes
        self.mean = mean
        self.std = std
        
        self.normalize = T.Normalize(mean = self.mean, std = self.std)
        
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(11, 11)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
        )
        
        self.features = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 64, kernel_size=(5,5)),
            nn.ReLU(),
        )
        
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(141376, 1000),
            nn.ReLU(),
            nn.Linear(1000, 64),
            nn.ReLU(),
            nn.Linear(64, out_features=self.num_classes)
        )
        
    def forward(self, X: torch.tensor) -> torch.tensor:
        X = self.normalize(X)
        X = self.input_layer(X)
        X = self.features(X)
        X = self.output(X)
        return X