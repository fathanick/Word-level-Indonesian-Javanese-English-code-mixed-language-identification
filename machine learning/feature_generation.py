from helper.dataset_reader import read_tsv
from helper.data_transformer import *
import re

def feature_extraction(tokens):
    features = []
    symbols = ['@', '#', '&', '$']

    # token features
    for index, token in enumerate(tokens):
        feature = {
            'token.lower': token.lower(),
            'n_gram_0': token,
            'token_BOS': index == 0,
            'token_EOS': index == len(tokens) - 1,
            'token.prefix_2': token[:2],
            'token.prefix_3': token[:3],
            'token.suffix_2': token[-2:],
            'token.suffix_3': token[-3:],
            'token.length': len(token),
            'token.is_alpha': token.isalpha(),
            'token.is_numeric': token.isnumeric(),
            'token.is_capital': token.isupper(),
            'token.is_title': token.istitle(),
            'token.startswith_symbols': (any(token.startswith(x) for x in symbols)),
            'token.contains_numeric': bool(re.search('[0-9]', token)),
            'token.contains_capital': (any(letter.isupper() for letter in token)),
            'token.contains_quotes': ('\"' in token) or ('\'' in token),
            'token.contains_hyphen': '-' in token,
        }

        features.append(feature)

    return features


if __name__ == "__main__":
    train_data = read_tsv("../dataset/16-07-22/train.tsv")
    test_data = read_tsv("../dataset/16-07-22/test.tsv")

    X_train = train_data[1]
    y_train = train_data[2]
    X_test = test_data[1]
    y_test = test_data[2]

    train_features = feature_extraction(X_train)
    df_train_features = pd.DataFrame(train_features)
    #df_train_features['tag'] = y_train
    df_train_features.to_csv('../dataset/16-07-22/train_features.txt', index=None, header=False, sep="\t")

    test_features = feature_extraction(X_test)
    df_test_features = pd.DataFrame(test_features)
    #df_test_features['tag'] = y_test
    df_test_features.to_csv('../dataset/16-07-22/test_features.txt', index=None, header=False, sep="\t")
