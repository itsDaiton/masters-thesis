from transformers import CLIPImageProcessor, CLIPTokenizer, AutoImageProcessor

model_names = {
    "ViT": "google/vit-base-patch16-224",
    "DeiT": "facebook/deit-base-distilled-patch16-224",
    "Swin": "microsoft/swin-base-patch4-window7-224",
    "RegNet": "facebook/regnet-y-160",
    "CLIP": "openai/clip-vit-base-patch16",
}


def get_last_layer(model, architecture):
    if architecture == "deit":
        return model.backbone.cls_classifier
    if architecture == "regnet":
        return model.backbone.classifier[1]
    if architecture == "clip":
        return model.model.classifier
    return model.backbone.classifier


def get_model_params(model):
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


def get_clip_processor_and_tokenizer(model_name="openai/clip-vit-base-patch16"):
    processor = CLIPImageProcessor.from_pretrained(model_name)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    return processor, tokenizer


def get_backbone_processor(model_name):
    if model_name == model_names["ViT"]:
        return AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    return AutoImageProcessor.from_pretrained(model_name)
