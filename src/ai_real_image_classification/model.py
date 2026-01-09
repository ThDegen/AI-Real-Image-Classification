import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, n_class=1, p_drop=0.5, pretrained=True, model_name='resnet18'):
        super().__init__()
        model = getattr(models, model_name)(weights='DEFAULT' if pretrained else None)

        # Backbone without the final FC layer
        self.backbone = nn.Sequential(*list(model.children())[:-1])

        # Classification head
        self.fc = nn.Sequential(
            nn.Dropout(p=p_drop),
            nn.Linear(model.fc.in_features, n_class)
        )

    def forward(self, x):
        x = self.backbone(x)          
        x = torch.flatten(x, 1)      
        x = self.fc(x)                
        return x
