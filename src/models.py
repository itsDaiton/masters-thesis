from transformers import ViTForImageClassification, DeiTForImageClassificationWithTeacher, RegNetForImageClassification, SwinForImageClassification
import torch.nn as nn

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
        return outputs.logits
    
class RegNet(nn.Module):
    """ RegNetX-4GF model pre-trained on ImageNet-1k. This model serves as a teacher model for DeiT. """
    
    def __init__(self, num_classes, model_name='facebook/regnet-x-040'):
        super(RegNet, self).__init__()
        self.backbone = RegNetForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        
    def forward(self, x):
        outputs = self.regnet(x)
        return outputs.logits
    
class Swin(nn.Module):
    """ Swin Transformer model pre-trained on ImageNet-1k. """
    
    def __init__(self, num_classes, model_name='microsoft/swin-tiny-patch4-window7-224'):
        super(Swin, self).__init__()
        self.backbone = SwinForImageClassification.from_pretrained(
            model_name, 
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        self.id2label = {i: f"label_{i}" for i in range(num_classes)}
        self.label2id = {f"label_{i}": i for i in range(num_classes)}
        
    def forward(self, x):
        outputs = self.backbone(x)
        return outputs.logits  