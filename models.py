import torch.nn as nn
from torchvision import models
from utils import freeze_params


def create_vision_model(model_type, num_classes, freeze, identity=False):

    if model_type=="resnet":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if freeze: freeze_params(model)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes) if not identity else nn.Identity()
        
    elif model_type=="mobilenet":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        if freeze: freeze_params(model)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes) if not identity else nn.Identity()

    elif model_type=="efficientnet2s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        if freeze: freeze_params(model)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes) if not identity else nn.Identity()

    elif model_type=="efficientnet":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        if freeze: freeze_params(model)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes) if not identity else nn.Identity()

    elif model_type=="efficientnet1":
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        if freeze: freeze_params(model)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes) if not identity else nn.Identity()

    elif model_type=="efficientnet2":
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        if freeze: freeze_params(model)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes) if not identity else nn.Identity()

    elif model_type=="efficientnet3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        if freeze: freeze_params(model)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes) if not identity else nn.Identity()

    elif model_type=="efficientnet4":
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        if freeze: freeze_params(model)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes) if not identity else nn.Identity()
    
    
    elif model_type=="convnext":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        if freeze: freeze_params(model)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, num_classes) if not identity else nn.Identity()

    elif model_type=="swint":
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        if freeze: freeze_params(model)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, num_classes) if not identity else nn.Identity()

    elif model_type=="swins":
        model = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)
        if freeze: freeze_params(model)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, num_classes) if not identity else nn.Identity()
        
    elif model_type=="swin2t":
        model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)
        if freeze: freeze_params(model)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, num_classes) if not identity else nn.Identity()

    elif model_type=="vitb16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        if freeze: freeze_params(model)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes) if not identity else nn.Identity()

    elif model_type=="vitb32":
        model = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
        if freeze: freeze_params(model)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes) if not identity else nn.Identity()

    else:
        assert False, f"Model '{model_type}' not implemented"

    if identity:
        return model, num_ftrs
    else:
        return model


class MultiTaskVisionModel(nn.Module):
    def __init__(self, model_type, num_diseases=5, num_symptoms=8, freeze=False):
        super(MultiTaskVisionModel, self).__init__()
        self.model_type = model_type

        base, in_features = create_vision_model(model_type, 0, freeze=freeze, identity=True)
        self.backbone = base
            
        # Build the Multi-Task Classification Heads
        self.disease_head = nn.Linear(in_features, num_diseases)
        self.symptom_head = nn.Linear(in_features, num_symptoms)

    def forward(self, x):
        # Extract joint visual embeddings from shared backbone
        features = self.backbone(x)
        
        # Branch independently into multi-label logits
        disease_logits = self.disease_head(features)
        symptom_logits = self.symptom_head(features)
        
        return disease_logits, symptom_logits


class S2D_MultiTaskVisionModel(MultiTaskVisionModel):

    def __init__(self, model_type, num_diseases=5, num_symptoms=8, freeze=False):
        super(S2D_MultiTaskVisionModel, self).__init__(model_type, num_diseases, num_symptoms, freeze)
        self.model_type = model_type

        self.disease_to_symptom_head = nn.Linear(num_diseases, num_symptoms)
        self.symptom_to_disease_head = nn.Linear(num_symptoms, num_diseases)

    def forward(self, x):
        # Extract joint visual embeddings from shared backbone
        features = self.backbone(x)
        
        # Branch independently into multi-label logits
        disease_logits = self.disease_head(features)
        symptom_logits = self.symptom_head(features)

        disease_logits2 = self.symptom_to_disease_head(symptom_logits)
        
        return disease_logits, symptom_logits, disease_logits2, None



