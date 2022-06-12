from helper.dataset_reader import read_tsv
from helper.data_transformer import *
from sklearn.model_selection import train_test_split

def split_dataset():
    # prepare data as input
    filename = '../dataset/comlid-data-140422-v1.tsv'
    data = read_tsv(filename)
    all_data, all_words, all_tags = data

    # transform data to dataframe
    df = to_df_tweet_tags(all_data)
    # print(df.head())

    # define X and y before splitting
    X = df['Tweets']
    y = df['Tags']

    # data splitting
    X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train_and_val, y_train_and_val, test_size=0.2, random_state=0)

    # create files: train.tsv, val.tsv, and test.tsv
    list_to_tsv(filename="../dataset/train.tsv", X=X_train, y=y_train)
    list_to_tsv(filename="../dataset/val.tsv", X=X_val, y=y_val)
    list_to_tsv(filename="../dataset/test.tsv", X=X_test, y=y_test)

if __name__ == "__main__":
    split_dataset()


