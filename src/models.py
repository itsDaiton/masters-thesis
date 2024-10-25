from transformers import AutoModelForImageClassification, CLIPModel
from utils.models_utils import model_names
import torch
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
    
    def __init__(self, model_name=model_names['CLIP'], for_training=False, num_classes=None, use_coop=False, num_prompt_tokens=None):
        super(CLIP, self).__init__()
        self.for_training = for_training
        self.use_coop = use_coop
        self.num_prompt_tokens = num_prompt_tokens
        
        if for_training and num_classes is not None:
            self.model = AutoModelForImageClassification.from_pretrained(
                pretrained_model_name_or_path=model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            self.model = CLIPModel.from_pretrained(pretrained_model_name_or_path=model_name)
                             
        if use_coop:
            embedd_dim = self.model.text_projection.in_features
            self.prompt_embedd = nn.Parameter(
                torch.randn(self.num_prompt_tokens, embedd_dim), 
                requires_grad=True
            )
 
    def forward(self, images, texts=None, labels=None):
        if self.use_coop and self.for_training is None and labels is not None:
            prompt_texts = []
            for label in labels:
                label_embedding = self.model.get_text_features(input_ids=label)
                prompt_embedding = torch.cat([self.prompt_embedd, label_embedding], dim=0)
                prompt_texts.append(prompt_embedding)
                
            text_features = torch.stack([self.model.get_text_features(text) for text in prompt_texts])
            
            image_features = self.model.get_image_featutures(images)
            logits_per_image = (image_features @ text_features.T) / self.model.logit_scale.exp()
            return logits_per_image
        
        elif self.for_training and not self.use_coop:
            outputs = self.model(pixel_values=images)
            return outputs.logits
        else:
            outputs = self.model(pixel_values=images, input_ids=texts)
            return outputs.logits_per_image