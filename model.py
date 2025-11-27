import torch
import torch.nn as nn
import torchvision.models as models

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Match training notebook EXACTLY
        backbone = models.efficientnet_b0(pretrained=True)
        feat_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        feats = self.backbone(x)

        # Your training notebook did this!
        if feats.ndim == 4:
            feats = torch.flatten(feats, 1)

        return self.head(feats).squeeze(1)
