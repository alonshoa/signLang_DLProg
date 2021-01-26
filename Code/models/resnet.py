import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
from torchsummary import summary
from singLang_DLProg.Code.data.dataloaders import get_dataloader


class SingLangResNet(ResNet):
    def __init__(self):
        super(SingLangResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=25)
        self.conv1 = torch.nn.Conv2d(3, 64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3), bias=False)


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y_hat = self.fc(x)

        return x,y_hat


if __name__ == '__main__':
    test_path = 'D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\debug'
    dl = get_dataloader(test_path)
    model = SingLangResNet()
    x,y = iter(dl).next()
    # print(x.shape)

    lat_x, y_hat = model(x)
    print("lat_x",lat_x.shape)
    print("y_hat",y_hat.shape)

    # summary(model.cuda(), input_size=(3, 224, 224))


    # print(y_hat)
