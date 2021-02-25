import torch
from Code.models.resnet import SingLangResNet
from torchvision import models

import os
dirname = os.path.dirname(__file__)
singLang_DLProg = os.path.join(dirname, '..', '..')
model = os.path.join(singLang_DLProg, 'pretrained', 'final_resnet_test_run_64.pt')

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



def get_pretrained_resnet(pretrained=True,num_classes=23):
    model=models.resnet18(pretrained)
    model.fc = torch.nn.Linear(512,num_classes)

    return model

def load_resnet_model(model_name=model):
    model = SingLangResNet()
    if model_name is not "":
        model.load_state_dict(
        torch.load(model_name, map_location='cpu'))
    return model

def save_model(model,path):
    torch.save(model.state_dict(), path)