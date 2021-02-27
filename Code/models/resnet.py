import torch
from torchvision.models.resnet import ResNet, BasicBlock
from torchsummary import summary



class SingLangResNet(ResNet):
    def __init__(self,num_channels=3,num_classes=23):
        super(SingLangResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.conv1 = torch.nn.Conv2d(num_channels, 64,
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
        y_hat = torch.softmax(self.fc(x),dim=1)

        return x,y_hat


if __name__ == '__main__':
# model summary
    model = SingLangResNet()
    summary(model.cuda(), input_size=(3, 224, 224))


