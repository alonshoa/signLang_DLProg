from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from singLang_DLProg.Code.utils.get_image_size import get_image_size

import re



def check_valid(filename):
    t = get_image_size(filename)
    # print(t)
    return t[0] == 224 and t[1] == 224

def get_dataloader(path_to_data, batch_size=16):
    dataset = ImageFolder(root=path_to_data, transform=transforms.ToTensor(),is_valid_file=check_valid)
    print(len(dataset.classes))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader

