from transformers import ViTForImageClassification, DeiTForImageClassificationWithTeacher
import torch.nn as nn
from torchvision.models.regnet import regnet_y_16gf, RegNet_Y_16GF_Weights

class ViT(nn.Module):
    """ Vision Transformer (ViT) model pre-trained on ImageNet-1k. """
    
    def __init__(self, num_classes, model_name='facebook/deit-small-patch16-224'):
        super(ViT, self).__init__()
        self.backbone = ViTForImageClassification.from_pretrained(
            model_name, 
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        
    def forward(self, x):
        outputs = self.backbone(x)
        return outputs.logits
    
class DeiT(nn.Module):
    """ Data-efficient Image Transformer (DeiT) model pre-trained on ImageNet-1k. """
    
    def __init__(self, num_classes, model_name='facebook/deit-small-distilled-patch16-224'):
        super(DeiT, self).__init__()
        self.backbone = DeiTForImageClassificationWithTeacher.from_pretrained(
            model_name, 
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        
    def forward(self, x):
        outputs = self.backbone(x)   
        class_logits = outputs.logits
        distillation_logits = outputs.distillation_logits
        return class_logits, distillation_logits
    
class RegNet(nn.Module):
    """ RegNetY-16GF model pre-trained on ImageNet-1k. This model serves as a teacher model for DeiT. """
    
    def __init__(self, num_classes):
        super(RegNet, self).__init__()
        self.regnet = regnet_y_16gf(weights=RegNet_Y_16GF_Weights.DEFAULT)
        
    def forward(self, x):
        return self.regnet(x)