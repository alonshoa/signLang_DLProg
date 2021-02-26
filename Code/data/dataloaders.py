from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from Code.utils.get_image_size import get_image_size
import torchvision
import torch

def check_valid(filename):
    t = get_image_size(filename)
    return t[0] == 224 and t[1] == 224



def get_image_dataloader(path_to_data, batch_size=16):
    my_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.RandomRotation([-30,30]),
        # transforms.Normalize([],[]) # get mean and std
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(root=path_to_data, transform=my_transforms,is_valid_file=check_valid)
    print(len(dataset.classes))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader,dataset


def get_mnist():
    from torchvision.datasets.mnist import MNIST
    dataset = MNIST('D:\\Alon_temp\\singlang\\singLang_DLProg\\out_puts', transform=torchvision.transforms.ToTensor(),download=True)
    print(len(dataset.classes))
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
    return dataloader


if __name__ == '__main__':
    test_path = 'D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\debug'
    dl = get_image_dataloader(test_path, batch_size=1)
    x, y = iter(dl).next()
    print("-----")
    print(x.shape)
    import matplotlib.pyplot as plt
    plt.imshow(x.squeeze(0).permute(1,2,0))
    plt.show()
    print(torch.equal(x[0,0],x[0,1]))
    print(torch.equal(x[0,0],x[0,2]))
    print(x[0,0] - x[0,1])
    print("-----")
