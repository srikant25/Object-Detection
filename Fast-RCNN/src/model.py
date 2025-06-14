import torch
from torchvision import models
from torchvision.ops import RoIPool
import torch.nn as nn

class FastRCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(FastRCNN, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        features = vgg16.features[:-1]  # Remove last maxpool layer
        self.backbone = nn.Sequential(*features)
        self.roi_pool = RoIPool(output_size=(7, 7), spatial_scale=1 / 16)
        self.fc6 = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.fc7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.classifier = nn.Linear(4096, num_classes)
        self.regressor = nn.Linear(4096, 4)

    def forward(self, image, roi):
        """
        image : Tensor [N, 3, H, W]
        roi   : Tensor [R, 5] --> [batch_idx, x1, y1, x2, y2]
        """
        feature_map = self.backbone(image)
        roi_map = self.roi_pool(feature_map, roi)
        pooled_features = torch.flatten(roi_map, start_dim=1)
        x = self.fc6(pooled_features)
        x = self.fc7(x)
        class_logits = self.classifier(x)
        bbox_pred = self.regressor(x)
        return class_logits, bbox_pred

    def predict(self, images, rois):
        """
        Inference mode — returns probabilities and raw bbox deltas
        """
        self.eval()
        with torch.no_grad():
            logits, bbox_deltas = self.forward(images, rois)
            probs = torch.softmax(logits, dim=1)
        return probs, bbox_deltas
