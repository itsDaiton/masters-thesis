from transformers import AutoModelForImageClassification, CLIPModel
from utils.models_utils import model_names
import torch.nn as nn

class Backbone(nn.Module):
    """ Base class for all backbone models used - ViT, DeiT, Swin, RegNet (for distillation). """
    
    def __init__(self, model_name, num_classes):
        super(Backbone, self).__init__()
        self.backbone = AutoModelForImageClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        
    def forward(self, x):
        outputs = self.backbone(x)
        return outputs.logits 
    
class CLIP(nn.Module):
    """ CLIP model pre-trained on YFCC100M dataset. """
    
    def __init__(self, model_name=model_names['CLIP'], for_training=False, num_classes=None, linear_probe=False):
        super(CLIP, self).__init__()
        self.for_training = for_training
        self.linear_probe = linear_probe
        
        if for_training and num_classes is not None:
            self.model = AutoModelForImageClassification.from_pretrained(
                pretrained_model_name_or_path=model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            self.model = CLIPModel.from_pretrained(pretrained_model_name_or_path=model_name)
 
    def forward(self, images, texts=None):   
        if self.for_training:
            outputs = self.model(pixel_values=images)
            return outputs.logits
        elif self.linear_probe:
            outputs = self.model(pixel_values=images, input_ids=texts)
            return outputs.image_embeds, outputs.text_embeds
        else:
            outputs = self.model(pixel_values=images, input_ids=texts)
            return outputs.logits_per_image
        
class LinearClassifier(nn.Module):
    """ Linear classifier trained on top of CLIP's image and text embeddings. """
    
    def __init__(self, in_features, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.fc(x)