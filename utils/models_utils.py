model_names = {
    'ViT': 'facebook/deit-small-patch16-224',
    'DeiT': 'facebook/deit-small-distilled-patch16-224',
    'Swin': 'microsoft/swin-tiny-patch4-window7-224',
    'RegNet': 'facebook/regnet-x-040',
    'CLIP': 'openai/clip-vit-base-patch16',
}

def get_last_layer(model, architecture):
    if architecture == 'deit':
        return model.backbone.cls_classifier
    elif architecture == 'regnet':
        return model.backbone.classifier[1]
    elif architecture == 'clip':
        return model.model.classifier
    else:
        return model.backbone.classifier
    
def get_model_params(model):
    print(f'Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
