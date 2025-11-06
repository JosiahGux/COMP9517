
import torch, torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
class ViTClassifier(nn.Module):
    def __init__(self, num_classes=12, pretrained=True):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.model = vit_b_16(weights=weights)
        in_f = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_f, num_classes)
    def forward(self, x):
        return self.model(x)
