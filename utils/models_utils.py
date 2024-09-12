model_names = {
    'ViT': 'facebook/deit-small-patch16-224',
    'DeiT': 'facebook/deit-small-distilled-patch16-224',
    'Swin': 'microsoft/swin-tiny-patch4-window7-224',
    'RegNet': 'facebook/regnet-x-040',
}

def get_last_layer(model, model_type):
    if model_type == 'deit':
        return model.backbone.cls_classifier
    elif model_type == 'regnet':
        return model.backbone.classifier[1]
    else:
        return model.backbone.classifier