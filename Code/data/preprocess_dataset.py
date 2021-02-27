"""'
we will use streets names as a dataset.
in this file we will do the preprocessing for israeli street-names. taken from ____

for the preprocess we will take the street names, text file containing only the names as a list

e.g.
תירוש
שדות מיכה
גפן
שקד
השקד

"""
import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_train_test(file_name,out_path):
    df = pd.read_csv(file_name, header=None,names=['street_name'])
    df['chars'] = df['street_name'].apply(lambda x: " ".join(x))
    train, test = train_test_split(df,test_size=0.1,random_state=42)
    train.to_csv(os.path.join(out_path,"train_.csv"),index=False)
    test.to_csv(os.path.join(out_path,"test_.csv"),index=False)
    # print(train)


def create_streets_only_csv(file_name, out_path):
    df = pd.read_csv(file_name)
    results = df.apply(lambda x: x['street_name'].strip(),axis=1)
    results.to_csv(out_path,index=False,header=False)
    print(results.value_counts())


def compute_mean_and_std(loader):
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)


if __name__ == '__main__':
    pass
    # # create csv file
    # file_name = '../../../singLang_DLProg/text_data/rechovot_2_20190501.csv'
    # out_path = '../../../singLang_DLProg/text_data/processed.csv'
    # print(os.getcwd(), file_name)
    #
    # create_streets_only_csv(file_name,out_path)

    # train\test split
    # file_name = '../../../singLang_DLProg/text_data/processed.csv'
    # out_path = '../../../singLang_DLProg/text_data/split'
    # split_train_test(file_name,out_path)
    #

    # calc stats from data
    # from Code.data.heb_data import create_street_names_data_iterators
    from Code.data.dataloaders import get_image_dataloader
    path = "D:\\Alon_temp\\singlang\\singLang_DLProg\\images\\coloredCaptureData"
    dataloader = get_image_dataloader(path)
    mean, std = compute_mean_and_std(dataloader)
    print("mean=",mean)
    print("std=",std)
