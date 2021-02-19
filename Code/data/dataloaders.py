from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from Code.utils.get_image_size import get_image_size
import torchvision
import torch

def check_valid(filename):
    t = get_image_size(filename)
    return t[0] == 224 and t[1] == 224

def get_dataloader(path_to_data, batch_size=16):
    dataset = ImageFolder(root=path_to_data, transform=transforms.ToTensor(),is_valid_file=check_valid)
    print(len(dataset.classes))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_mnist():
    from torchvision.datasets.mnist import MNIST
    dataset = MNIST('D:\\Desktop\\ofek\\STUDY\\university\\year6\\deepLearning\\final_project\\singLang_DLProg\\out_puts', transform=torchvision.transforms.ToTensor(),
                    download=True)
    print(len(dataset.classes))
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
    return dataloader


if __name__ == '__main__':
    test_path = 'D:\\Desktop\\ofek\\STUDY\\university\\year6\\deepLearning\\final_project\\images\\debug'
    dl = get_dataloader(test_path,batch_size=1)
    x, y = iter(dl).next()
    # print(x.shape)
    a = x[0,0] - x[0,2]
    b = x[0, 0] - x[0, 1]
    print(torch.unique(a))
    print(torch.unique(b))
    print("-----")
    print(y.shape)
    print(y)
    print("-----")
