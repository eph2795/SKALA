import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead


class MaskRCNN(nn.Module):
    def __init__(self, num_classes=21):
        super(MaskRCNN, self).__init__()

        self.model = maskrcnn_resnet50_fpn(weights=None)
        self._modify_anchors()
        # Replace the mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes,
        )
        for param in self.model.backbone.parameters():
            param.requires_grad = False

    def _modify_anchors(self):
        """Модифицирует anchor boxes"""

        # Anchor boxes для гранул
        anchor_sizes = (
            (8, 16, 32),
            (32, 48, 64),
            (64, 96, 128),
            (128, 192, 256),
            (256, 384, 512),
        )  # (8, 16, 32),
        aspect_ratios = ((0.7, 1.0),) * len(anchor_sizes)  # , 1.4

        # Создаем новый AnchorGenerator
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        num_anchors_per_location = anchor_generator.num_anchors_per_location()[0]

        # Получаем размер входных признаков для RPN
        in_channels = 256  # для FPN выход всегда 256 каналов

        # Создаем новый RPN head с правильным количеством anchor boxes
        rpn_head = RPNHead(in_channels=in_channels, num_anchors=num_anchors_per_location)

        # Заменяем anchor generator
        self.model.rpn.anchor_generator = anchor_generator
        self.model.rpn.head = rpn_head

        # print("Anchor boxes установлены")
        # print(f"  Anchors per location: {num_anchors_per_location}")

    def forward(self, x, targets=None):
        if self.training:
            return self.model(x, targets)
        else:
            return self.model(x)
