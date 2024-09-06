from transformers import ViTForImageClassification
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