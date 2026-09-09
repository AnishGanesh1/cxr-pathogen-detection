import torch
import torch.nn as nn
from torchvision import models

def get_mimic_model(num_classes):
    # Use DenseNet-121 as a backbone for classification
    # Better for medical imaging (captures dense feature maps)
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    
    # Replace the final classifier layer
    # DenseNet has .classifier instead of .fc
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
        # Sigmoid removed - use BCEWithLogitsLoss during training
    )
    
    return model
