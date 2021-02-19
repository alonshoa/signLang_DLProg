import pandas as pd
from torch import nn
from torch.utils.data import dataloader


class HebDataloader(dataloader.DataLoader):
    def __init__(self,root_dir,train=True,test=True):
        assert train or test # load atleast one dataset
        if train:
            self.load_train(root_dir)
        if test:
            self.load_test(root_dir)

    def load_train(self, root_dir):
        pass

    def load_test(self, root_dir):
        pass


if __name__ == '__main__':
    df = pd.read_csv('/singLang_DLProg/text_data/rechovot_2_20190501.csv')
    print(df.columns)
    for i in range(5):
        print(df.iloc[i]['street_name'].strip())
    print(df.head())
