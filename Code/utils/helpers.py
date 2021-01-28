import torch
from singLang_DLProg.Code.models.resnet import SingLangResNet


def load_resnet_model(model_name='D:\\Alon_temp\\singlang\\singLang_DLProg\\out_puts\\resnet_test_run_64_10000.pt'):
    model = SingLangResNet()
    model.load_state_dict(
    torch.load(model_name))
    return model
