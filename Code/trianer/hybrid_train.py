import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Code.data.dataloaders import get_image_dataloader
from Code.data.heb_data import create_street_names_data_iterators
from Code.data.hybrid_data import HybridDataSet
from Code.models.heb_model import HebLetterToSentence
from Code.models.hybrid_model import HybridModel
from Code.trianer.train_images import Trainer
from Code.utils.helpers import load_resnet_model, set_grads_to_false


class HybridTrainer(Trainer):
    def __init__(self, train_dataloader, eval_dataloader, model, optim, summary_writer,
                 criteria=torch.nn.CrossEntropyLoss().cuda(), save_checkpoint=200,device='cuda'):
        super(HybridTrainer, self).__init__(train_dataloader, eval_dataloader, model, optim, summary_writer,
                                          criteria=criteria, save_checkpoint=save_checkpoint)

        self.device = device


    def train_epoch(self):
        avg_loss = 0
        for x, y in tqdm(self.train_dataloader):
            self.model.train()
            self.optim.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            y_hat, _ = self.model(x)
            loss = self.criteria(y_hat.squeeze(1), y)
            avg_loss += loss.item()
            loss.backward()
            self.optim.step()
        return avg_loss / len(self.train_dataloader)



    def log_epoch(self, acc_log, avg_loss, device, epoch, loss_log, summary_writer):
        # display
        # acc_test = accuracy(self.model, self.eval_dataloader, device)
        # acc_train = accuracy(self.model, self.train_dataloader, device)
        print(f"loss[{epoch}] = {avg_loss / self.train_size}")
        # print(f"acc[{epoch}] = {acc_test.item()}")
        loss_log.append(avg_loss / self.train_size)
        summary_writer.add_scalar("loss_train_text", avg_loss, global_step=epoch)
        # summary_writer.add_scalar("accuracy_test", acc_test.item(), global_step=epoch)
        # summary_writer.add_scalar("accuracy_train",acc_train.item())
        # acc_log.append(acc_test)


if __name__ == '__main__':
    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    path_to_data = "D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\coloredCaptureData\\train"
    _,imageDataSet = get_image_dataloader(path_to_data, batch_size=16)

    # path = "C:\\HW\\singLang_DLProg\\text_data\\split"
    path = "D:\\Alon_temp\\singlang\\singLang_DLProg\\text_data\\split"

    train_iterator, test_iterator,vocab_letters,vocab_words,train_data,test_data  = create_street_names_data_iterators(path)

    hybridDataSet = HybridDataSet(imageDataSet,test_data,vocab_words)
    dataloader = DataLoader(dataset=hybridDataSet, batch_size=4, shuffle=True)
    # someItem = hybridDataSet.__getitem__(2)
    # print(someItem[0].shape)
    # print(hybridDataSet.word_max_len)
    # def __init__(self, vocab_size,embedding_dim, lstm_size,hidden_dim, output_dim):

    text_model = HebLetterToSentence(len(vocab_letters), 128, 512, 128,len(vocab_words),use_self_emmbed=True)
    image_model = load_resnet_model('D:\\Alon_temp\\singlang\\singLang_DLProg\\out_puts\\final_resnet_with_aug_colored_pretrained_test_run_64_trainer.pt')

    image_model = set_grads_to_false(image_model)
    # exit(12)
    model = HybridModel(image_model,text_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    writer = SummaryWriter(comment="full_train_pretrained_resnet_no_image_update")
    # print(model)
    # x,y = next(iter(dataloader))
    # print(model(x))
    trainer = HybridTrainer(dataloader, test_iterator, model, optimizer, writer,save_checkpoint=5,device=device)

    trainer.train("full_run_test_no_image_update",epochs=100, device=device)



# import os
# from random import uniform, random, choice, sample
# import os
# # from torchtext.data import Field, BucketIterator, TabularDataset
# # from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
# import torch
# from torch.utils.data import DataLoader,Dataset
# from Code.data.dataloaders import get_image_dataloader
# from Code.data.heb_data import create_street_names_data_iterators
# from Code.models.resnet import SingLangResNet
# import matplotlib.pyplot as plt
#
# class HybridDataSet(Dataset):
#
#     def __init__(self,imageDataSet,textDataSet):
#         self.textDataSet = textDataSet
#         self.imageDataSet = imageDataSet
#         self.labelImageIndexDic = self.createletterToIndexDic()
#         self.word_max_len = self.get_max_len()
#     '''
#     create mapping between letter to the matching image index in the dataset
#     '''
#     def createletterToIndexDic(self):
#         dic ={}
#         max_size = -1
#         for i in range(len(self.imageDataSet)):
#             image,letter = self.imageDataSet[i]
#             if letter in dic.keys():
#                 dic[self.imageDataSet.classes[letter]].append(i)
#             else:
#                 dic[self.imageDataSet.classes[letter]] = [i]
#         return dic
#
#     def __len__(self):
#         return len(self.textDataSet)
#
#     def __getitem__(self, item):
#         textItem = self.textDataSet[item]
#         imagesIndexes = []
#         for hebLetter in textItem.chars:
#             engLetter = self.hebToEngConvertor(hebLetter)
#             imageIndexes = self.labelImageIndexDic[engLetter]
#             randomImage = choice(imageIndexes)
#
#             imagesIndexes.append(randomImage)
#
#         # add padding
#         for i in range(self.word_max_len - len(textItem.chars)):
#             engLetter = self.hebToEngConvertor('pad')
#             imageIndexes = self.labelImageIndexDic[engLetter]
#             randomImage = choice(imageIndexes)
#
#             imagesIndexes.append(randomImage)
#             # image, letter = self.imageDataSet[randomImage]
#             # plt.imshow(image.squeeze(0).permute(1,2,0).numpy())
#             # print(hebLetter)
#             # plt.show()
#         # imagesFromDataSet = []
#         # for index in imagesIndexes:
#         #     images, letters = self.imageDataSet[index]
#         #     print(images.shape)
#         #     break
#         #     imagesFromDataSet.append(images)
#         imagesFromDataSet = torch.cat([self.imageDataSet[index][0].unsqueeze(0) for index  in imagesIndexes],dim=0)
#         return imagesFromDataSet,textItem.names
#
#     def hebToEngConvertor(self,x):
#         return {
#             'א': 'A',
#              'ב': 'B',
#             'ג': 'C',
#             'ד': 'D',
#             'ה': 'E',
#             'ו': 'F',
#             'ז': 'G',
#             'ח': 'H',
#             'ט': 'I',
#             'י': 'J',
#             'כ': 'K',
#             'ך': 'K',
#             'ל': 'L',
#             'מ': 'M',
#             'נ': 'N',
#             'ן': 'N',
#             'ס': 'O',
#             'ע': 'P',
#             'פ': 'Q',
#             'ף': 'Q',
#             'צ': 'R',
#             'ץ': 'R',
#             'ק': 'S',
#             'ר': 'T',
#             'ש': 'U',
#             'ת': 'V',
#             'pad': 'W',
#         }[x]
#
#     def get_max_len(self):
#         max_len = -1
#         for s in self.textDataSet:
#             if len(s.chars) > max_len:
#                 max_len = len(s.chars)
#         return max_len
#
#
#
# if __name__ == '__main__':
#     # path_to_data = "C:\HW\singLang_DLProg\images\coloredCaptureData_debug"
#     path_to_data = "D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\coloredCaptureData_debug"
#     _,imageDataSet = get_image_dataloader(path_to_data, batch_size=16)
#
#     # path = "C:\\HW\\singLang_DLProg\\text_data\\split"
#     path = "D:\\Alon_temp\\singlang\\singLang_DLProg\\text_data\\split"
#
#     train_iterator, test_iterator,_,_,train_data,test_data  = create_street_names_data_iterators(path)
#
#     hybridDataSet = HybridDataSet(imageDataSet,test_data)
#     dataloader = DataLoader(dataset=hybridDataSet, batch_size=16, shuffle=True)
#     someItem = hybridDataSet.__getitem__(2)
#     print(someItem[0].shape)
#     print(hybridDataSet.word_max_len)
