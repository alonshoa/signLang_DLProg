import torch
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet

from Code.models.resnet import SingLangResNet
from torchvision import models



import os
dirname = os.path.dirname(__file__)
singLang_DLProg = os.path.join(dirname, '..', '..')
model_name = os.path.join(singLang_DLProg, 'pretrained', 'final_resnet_test_run_64.pt')

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def get_pretrained_resnet(pretrained=True,embbeding_dim=512, num_classes=23):
    from torch.hub import load_state_dict_from_url
    # model=models.resnet18(pretrained)
    sign_resnet = SingLangResNet(num_classes=1000)
    state_dict = load_state_dict_from_url(model_urls['resnet18'])
    sign_resnet.load_state_dict(state_dict)
    sign_resnet.fc = torch.nn.Linear(embbeding_dim,num_classes)

    return sign_resnet


def load_resnet_model(model_name=model_name):
    model = SingLangResNet()
    if model_name is not "":
        model.load_state_dict(
        torch.load(model_name, map_location='cpu'))
    return model

def save_model(model,path):
    torch.save(model.state_dict(), path)


def set_grads_to_false(model):
    for param in model.parameters():
        param.requires_grad = False
    return model
