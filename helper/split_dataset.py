from helper.dataset_reader import read_tsv
from helper.data_transformer import *
from sklearn.model_selection import train_test_split
import os

def read_dataset(filename):
    data = read_tsv(filename)
    all_data, all_words, all_tags = data

    # transform data to dataframe
    df = to_df_tweet_tags(all_data)
    # print(df.head())

    # define X and y before splitting
    X = df['Tweets']
    y = df['Tags']

    return X, y

def split_train_val_test(X, y, root_path):

    # data splitting
    X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train_and_val, y_train_and_val, test_size=0.2, random_state=0)

    # create files: train.tsv, val.tsv, and test.tsv
    list_to_tsv(filename=os.path.join(root_path, 'train.tsv'), X=X_train, y=y_train)
    list_to_tsv(filename=os.path.join(root_path, 'val.tsv'), X=X_val, y=y_val)
    list_to_tsv(filename=os.path.join(root_path, 'test.tsv'), X=X_test, y=y_test)

def split_train_test(X, y, root_path):
    # data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    # create files: train.tsv, val.tsv, and test.tsv
    list_to_tsv(filename=os.path.join(root_path, 'train.tsv'), X=X_train, y=y_train)
    list_to_tsv(filename=os.path.join(root_path, 'test.tsv'), X=X_test, y=y_test)

if __name__ == "__main__":
    filename = '../dataset/ijelid-batch-1.tsv'
    root_path = '../dataset/split_batch_1/'
    X, y = read_dataset(filename)
    split_train_val_test(X, y, root_path)


