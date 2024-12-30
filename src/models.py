from transformers import AutoModelForImageClassification, CLIPModel
from torch import nn
from utils.models_utils import model_names


class Backbone(nn.Module):
    """Base class for all backbone models used - ViT, DeiT, Swin, RegNet (for distillation)."""

    def __init__(self, model_name, num_classes, dropout=False, dropout_rate=0.1):
        super(Backbone, self).__init__()
        self.backbone = AutoModelForImageClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        if dropout:
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )

    def forward(self, x):
        outputs = self.backbone(x)
        return outputs.logits


class CLIP(nn.Module):
    """CLIP model pre-trained on YFCC100M dataset."""

    def __init__(
        self,
        model_name=model_names["CLIP"],
        for_training=False,
        num_classes=None,
        dropout=False,
        dropout_rate=0.1,
    ):
        super(CLIP, self).__init__()
        self.for_training = for_training

        if for_training and num_classes is not None:
            self.model = AutoModelForImageClassification.from_pretrained(
                pretrained_model_name_or_path=model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            self.model = CLIPModel.from_pretrained(
                pretrained_model_name_or_path=model_name
            )
        if dropout:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )

    def forward(self, images, texts=None):
        if self.for_training:
            outputs = self.model(pixel_values=images)
            return outputs.logits
        outputs = self.model(pixel_values=images, input_ids=texts)
        return outputs.logits_per_image
