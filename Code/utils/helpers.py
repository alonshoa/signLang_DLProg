import torch
from Code.models.resnet import SingLangResNet

import os
dirname = os.path.dirname(__file__)
singLang_DLProg = os.path.join(dirname, '..', '..')
model = os.path.join(singLang_DLProg, 'pretrained', 'final_resnet_test_run_64.pt')

def load_resnet_model(model_name=model):
    model = SingLangResNet()
    model.load_state_dict(
    torch.load(model_name, map_location='cpu'))
    return model
