import os
from random import uniform, random, choice, sample
import os
# from torchtext.data import Field, BucketIterator, TabularDataset
# from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

from torch.utils.data import DataLoader,Dataset
from Code.data.dataloaders import get_image_dataloader
from Code.data.heb_data import create_street_names_data_iterators
from Code.models.resnet import SingLangResNet
import matplotlib.pyplot as plt

class HybridDataSet(Dataset):

    def __init__(self,imageDataSet,textDataSet):
        self.textDataSet = textDataSet
        self.imageDataSet = imageDataSet
        self.labelImageIndexDic = self.createletterToIndexDic()
    '''
    create mapping between letter to the matching image index in the dataset
    '''
    def createletterToIndexDic(self):
        dic ={}
        for i in range(len(self.imageDataSet)):
            image,letter = self.imageDataSet[i]
            if letter in dic.keys():
                dic[self.imageDataSet.classes[letter]].append(i)
            else:
                dic[self.imageDataSet.classes[letter]] = [i]
        return dic

    def __len__(self):
        return len(self.textDataSet)

    def __getitem__(self, item):
        textItem = self.textDataSet[item]

        imagesIndexes = []
        for hebLetter in textItem.chars:
            engLetter = self.hebToEngConvertor(hebLetter)
            imageIndexes = self.labelImageIndexDic[engLetter]
            randomImage = choice(imageIndexes)

            imagesIndexes.append(randomImage)
            # image, letter = self.imageDataSet[randomImage]
            # plt.imshow(image.squeeze(0).permute(1,2,0).numpy())
            # print(hebLetter)
            # plt.show()
        imagesFromDataSet = []
        for index in imagesIndexes:
            images, letters = self.imageDataSet[index]
            imagesFromDataSet.append(images)

        return imagesFromDataSet,textItem.names

    def hebToEngConvertor(self,x):
        return {
            'א': 'A',
             'ב': 'B',
            'ג': 'C',
            'ד': 'D',
            'ה': 'E',
            'ו': 'F',
            'ז': 'G',
            'ח': 'H',
            'ט': 'I',
            'י': 'J',
            'כ': 'K',
            'ך': 'K',
            'ל': 'L',
            'מ': 'M',
            'נ': 'N',
            'ן': 'N',
            'ס': 'O',
            'ע': 'P',
            'פ': 'Q',
            'ף': 'Q',
            'צ': 'R',
            'ץ': 'R',
            'ק': 'S',
            'ר': 'T',
            'ש': 'U',
            'ת': 'V',
        }[x]


if __name__ == '__main__':
    path_to_data = "C:\HW\singLang_DLProg\images\coloredCaptureData_debug"
    _,imageDataSet = get_image_dataloader(path_to_data, batch_size=16)

    path = "C:\\HW\\singLang_DLProg\\text_data\\split"
    train_iterator, test_iterator,_,_,train_data,test_data  = create_street_names_data_iterators(path)

    hybridDataSet = HybridDataSet(imageDataSet,test_data)
    dataloader = DataLoader(dataset=hybridDataSet, batch_size=16, shuffle=True)
    someItem = hybridDataSet.__getitem__(2)
    print()
