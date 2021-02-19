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
    train, test = train_test_split(df,test_size=0.1,random_state=42)
    train.to_csv(os.path.join(out_path,"train.csv"),index=False,header=False)
    test.to_csv(os.path.join(out_path,"test.csv"),index=False,header=False)
    # print(train)


def create_streets_only_csv(file_name, out_path):
    df = pd.read_csv(file_name)
    results = df.apply(lambda x: x['street_name'].strip(),axis=1)
    results.to_csv(out_path,index=False,header=False)
    print(results.value_counts())


if __name__ == '__main__':
    # # create csv file
    # file_name = '../../../singLang_DLProg/text_data/rechovot_2_20190501.csv'
    # out_path = '../../../singLang_DLProg/text_data/processed.csv'
    # print(os.getcwd(), file_name)
    #
    # create_streets_only_csv(file_name,out_path)

    # train\test split
    file_name = '../../../singLang_DLProg/text_data/processed.csv'
    out_path = '../../../singLang_DLProg/text_data/split'
    split_train_test(file_name,out_path)

