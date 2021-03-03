from torch.utils.data import DataLoader

from Code.data.dataloaders import get_image_dataloader
from Code.data.heb_data import create_street_names_data_iterators
from Code.data.hybrid_data import HybridDataSet
import torch.nn as nn

from Code.models.heb_model import HebLetterToSentence
from Code.utils.helpers import load_resnet_model


class HybridModel(nn.Module):
    def __init__(self,image_model,text_model):
        super(HybridModel, self).__init__()
        self.image_model = image_model
        self.text_model = text_model

    def forward(self,x):
        # x_last,y_hat = self.image_model(x) # [b,c,h,w]  ---  [b,chars,channels,h,w]
        bs = x.shape[0]
        x_last, y = self.image_model(x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1]))  # [b*chars,512]
        # print(x_last.shape)
        x_last = x_last.reshape(bs, -1, x_last.shape[-1]) # [b,chars,512]
        # print(x_last.shape)

        h0 = self.text_model.init_state(x_last.shape[1]) # chars
        # print("h0",h0.shape)
        out = self.text_model(x_last,h0.cuda())
        return out

if __name__ == '__main__':
    path_to_data = "D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\coloredCaptureData_debug"
    _,imageDataSet = get_image_dataloader(path_to_data, batch_size=16)

    # path = "C:\\HW\\singLang_DLProg\\text_data\\split"
    path = "D:\\Alon_temp\\singlang\\singLang_DLProg\\text_data\\split"

    train_iterator, test_iterator,vocab_letters,vocab_words,train_data,test_data  = create_street_names_data_iterators(path)

    hybridDataSet = HybridDataSet(imageDataSet,test_data)
    dataloader = DataLoader(dataset=hybridDataSet, batch_size=2, shuffle=True)
    # someItem = hybridDataSet.__getitem__(2)
    # print(someItem[0].shape)
    # print(hybridDataSet.word_max_len)
    # def __init__(self, vocab_size,embedding_dim, lstm_size,hidden_dim, output_dim):

    text_model = HebLetterToSentence(len(vocab_letters), 128, 512, 128,len(vocab_words),use_self_emmbed=True)
    image_model = load_resnet_model('')
    model = HybridModel(image_model,text_model)
    print(model)
    x,y = next(iter(dataloader))
    print(model(x))
