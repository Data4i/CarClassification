import torch
from torch import nn

class CarClassifierV0(nn.Module):
    def __init__(self, num_classes):
        super(CarClassifierV0, self).__init__()
        self.num_classes = num_classes
        
        self.input_layer = nn.Sequential([
            nn.Conv2d(224, 32, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
        ])
        
        self.features = nn.Sequential([
            nn.Conv2d(32, 64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 64, kernel_size=(3,3)),
            nn.ReLU(),
        ])
        
        self.output = nn.Sequential([
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes)
        ])
        
    def forward(self, X: torch.tensor) -> torch.tensor:
        x = self.input_layer(x)
        x = self.features(x)
        x = self.output(x)
        return x