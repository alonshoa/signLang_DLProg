import os
from torchtext.data import Field, BucketIterator, TabularDataset
import matplotlib.pyplot as plt

def create_street_names_data_iterators(path,char_max_size=50,names_max_size=1000,batch_size=32,device='cuda'):
    chars = Field(sequential=True, use_vocab=True, tokenize=lambda x: x.split(), lower=True)
    names = Field(sequential=False, use_vocab=True, tokenize=lambda x: x, lower=True)

    fields = {"chars": ("chars", chars), "street_name": ("names", names)}


    train_data, test_data = TabularDataset.splits(path="", train=os.path.join(path,"train_.csv"), test=os.path.join(path,"test_.csv"), format="csv", fields=fields)

    chars.build_vocab(train_data, max_size=char_max_size, min_freq=2)
    names.build_vocab(train_data, max_size=names_max_size, min_freq=1)

    # print(chars.vocab)
    # plt.hist(chars.vocab)
    # plt.show()
    # print(names.vocab.freqs)

    train_iterator, test_iterator = BucketIterator.splits(
        (train_data, test_data), batch_size=batch_size, device=device
)
    return train_iterator, test_iterator, chars.vocab, names.vocab

if __name__ == '__main__':
    path = "D:\\Alon_temp\\singlang\\singLang_DLProg\\text_data\\split"
    train_iterator, test_iterator,_,_ = create_street_names_data_iterators(path)
    # for batch in train_iterator:
    #     print(batch.chars)
    #     print(batch.names)
    #



