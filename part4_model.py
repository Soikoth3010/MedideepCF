# part4_model.py
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x) * x
        sa = self.spatial_attention(torch.cat([ca.mean(1, keepdim=True), ca.max(1, keepdim=True)[0]], dim=1))
        return sa * ca

class MediDeepCFNet(nn.Module):
    def __init__(self, num_classes=4):
        super(MediDeepCFNet, self).__init__()
        self.seg_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        self.seg_model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.cbam = CBAM(1280)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        seg_out = self.seg_model(x)['out']
        feats = self.efficientnet.features(x)
        feats = self.cbam(feats)
        class_out = self.classifier(feats)
        return seg_out, class_out
