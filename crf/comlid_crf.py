import random
import scipy.stats
import sklearn_crfsuite
import pickle
import os
import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import eli5
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, make_scorer
from sklearn_crfsuite import metrics
from helper.splitter import sentence_splitter
from warnings import simplefilter  # import warnings filter
from collections import Counter

simplefilter(action='ignore', category=FutureWarning)  # ignore all future warnings


class LanguageIdentifier:
    model = None
    window = 5
    symbols = ['@', '#', '&', '$']
    labels = ['ID', 'JV', 'EN', 'O', 'MIX-ID-EN', 'MIX-ID-JV', 'MIX-JV-EN']
    c1 = 0.1
    c2 = 0.1

    def __init__(self):
        random.seed(0)
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',  # for gradient descent for optimization and getting model parameters
            c1=0.1,  # Coefficient for Lasso (L1) regularization
            c2=0.1,  # Coefficient for Ridge (L2) regularization
            max_iterations=100,
            # The maximum number of iterations for optimization algorithms,
            # iteration for the gradient descent optimization
            all_possible_transitions=True
            # Specify whether CRFsuite generates transition features that do not even occur in the training data
        )

    def feature_extraction(self, tokens):
        features = []

        for index, token in enumerate(tokens):
            feature = {
                'n_gram_0': token,
                'token_BOS': index == 0,
                'token_EOS': index == len(tokens) - 1,
                'token.lower': token.lower(),
                'token.prefix_2': token[:2],
                'token.prefix_3': token[:3],
                'token.suffix_2': token[-2:],
                'token.suffix_3': token[-3:],
                'token.length': len(token),
                'token.is_alpha': token.isalpha(),
                'token.is_numeric': token.isnumeric(),
                'token.is_capital': token.isupper(),
                'token.is_title': token.istitle(),
                'token.startswith_symbols': (any(token.startswith(x) for x in self.symbols)),
                'token.contains_numeric': bool(re.search('[0-9]', token)),
                'token.contains_capital': (any(letter.isupper() for letter in token)),
                'token.contains_quotes': ('\"' in token) or ('\'' in token),
                'token.contains_hyphen': '-' in token,
            }

            features.append(feature)

        return features

    def token2features(self, sentence, index):
        # sentence: [w1, w2, ...], index: the index of the word
        # apply 22 features for each token

        token = sentence[index][0]

        features = {
            'n_gram_0': token,
            'token_BOS': index == 0,
            'token_EOS': index == len(sentence) - 1,
            'prev_tag': '' if index == 0 else sentence[index - 1][1],
            'next_tag': '' if index == len(sentence) - 1 else sentence[index + 1][1],
            'prev_2tag': '' if index == 0 or index == 1 else sentence[index - 2][1],
            'next_2tag': '' if index == len(sentence) - 1 or index == len(sentence) - 2 else sentence[index + 2][1],
            'prev_token': '' if index == 0 else sentence[index - 1][0],
            'next_token': '' if index == len(sentence) - 1 else sentence[index + 1][0],
            'token.lower': token.lower(),
            'token.prefix_2': token[:2],
            'token.prefix_3': token[:3],
            'token.suffix_2': token[-2:],
            'token.suffix_3': token[-3:],
            'token.length': len(token),
            'token.is_alpha': token.isalpha(),
            'token.is_numeric': token.isnumeric(),
            'token.is_capital': token.isupper(),
            'token.is_title': token.istitle(),
            'token.startswith_symbols': (any(token.startswith(x) for x in self.symbols)),
            'token.contains_numeric': bool(re.search('[0-9]', token)),
            'token.contains_capital': (any(letter.isupper() for letter in token)),
            'token.contains_quotes': ('\"' in token) or ('\'' in token),
            'token.contains_hyphen': '-' in token,
        }

        if len(token) > 5 and not (any(token.startswith(x) for x in self.symbols)):
            for i in range(0, len(token) - self.window):
                features[f'n_gram_{i}'] = token[i:(i + self.window)]

        return features

    def hyperparameter_optimization(self, data):
        data = self.data_transformer(data)

        X = [self.sent2features(s) for s in data]
        y = [self.sent2tags(s) for s in data]

        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
        }

        f1_scorer = make_scorer(metrics.flat_f1_score,
                                average='weighted',
                                labels=self.labels)

        rs = RandomizedSearchCV(self.model, params_space,
                                cv=5,
                                verbose=1,
                                n_jobs=-1,
                                n_iter=50,
                                scoring=f1_scorer)

        rs.fit(X, y)

        print('best params:', rs.best_params_)
        print('best CV score:', rs.best_score_)
        print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

        self.c1 = rs.best_params_['c1']
        self.c2 = rs.best_params_['c2']
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=100,
            all_possible_transitions=True,
            c1=self.c1,
            c2=self.c2,
        )

    @staticmethod
    def data_transformer(data):
        # input: [[[list of tokens],[list of tags]], [[list of tokens],[list of tags]]]
        # transform data to token and tag pairs: [[(tk1, tg1),(tk2, tg2)], [(tk1, tg1), (tk2, tg2)]]

        # 1. convert to dataframe
        df = pd.DataFrame(data[0], columns=['Tweets', 'Tags'])

        # 2. create token and tag pairs
        token_tag_pair = []
        for index, row in df.iterrows():
            pair = list(zip(row['Tweets'], row['Tags']))
            token_tag_pair.append(pair)

        return token_tag_pair

    def sent2features(self, sent):
        # get token features from token-tag pairs
        return [self.token2features(sent, i) for i in range(len(sent))]

    @staticmethod
    def sent2tags(sent):
        # get tags from token-tag pairs
        return [tag for token, tag in sent]

    def show_confusion_matrix(self, y, y_pred):

        # create a confusion matrix
        print('Confusion Matrix')
        flat_y = [item for y_ in y for item in y_]
        flat_y_pred = [item for y_pred_ in y_pred for item in y_pred_]
        cm = confusion_matrix(flat_y, flat_y_pred, labels=self.labels)
        # print(cm)

        # show classification report
        print(classification_report(flat_y, flat_y_pred, labels=self.labels))

        # sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax = plt.subplot()
        # sns.set(rc={'figure.figsize': (15, 12)})
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        # annot=True to annotate cells, ftm='g' to disable scientific notation

        # labels, title and ticks
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.xaxis.set_ticklabels(self.labels, rotation=90)
        ax.yaxis.set_ticklabels(self.labels, rotation=0)

        plt.show()

    def save_model(self, model_name):
        # save the model to disk
        root_path = '../model/'
        joined_path = os.path.join(root_path, model_name)
        pickle.dump(self.model, open(joined_path, 'wb'))

    @staticmethod
    def print_transitions(trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-10s -> %-10s %0.5f" % (label_from, label_to, weight))

    @staticmethod
    def print_state_features(state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-10s %s" % (weight, label, attr))

    def train_test_crf(self, X_train, y_train, X_test, y_test, model_name):
        # train the data
        self.model.fit(X_train, y_train)

        # prediction
        y_pred_test = self.model.predict(X_test)
        y_pred_train = self.model.predict(X_train)

        # show confusion matrix
        print('\n Evaluation on the test data')
        self.show_confusion_matrix(y_test, y_pred_test)
        print('\n Evaluation on the training data')
        self.show_confusion_matrix(y_train, y_pred_train)

        # show top likely and unlikely transitions
        print("\nTop likely transitions:")
        self.print_transitions(Counter(self.model.transition_features_).most_common(20))
        print("\nTop unlikely transitions:")
        self.print_transitions(Counter(self.model.transition_features_).most_common()[-20:])

        # check the state features
        print("\nTop positive:")
        self.print_state_features(Counter(self.model.state_features_).most_common(20))
        print("\nTop negative:")
        self.print_state_features(Counter(self.model.state_features_).most_common()[-20:])

        eli5.show_weights(self.model)
        # save model
        self.save_model(model_name)

    def pipeline_merge(self, data, test_size, model_name):
        data = self.data_transformer(data)

        X = [self.sent2features(s) for s in data]
        y = [self.sent2tags(s) for s in data]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        self.train_test_crf(X_train, y_train, X_test, y_test, model_name)

    def pipeline_split(self, train_data, test_data, model_name):
        train = self.data_transformer(train_data)
        test = self.data_transformer(test_data)

        X_train = [self.sent2features(s) for s in train]
        y_train = [self.sent2tags(s) for s in train]

        X_test = [self.sent2features(s) for s in test]
        y_test = [self.sent2tags(s) for s in test]

        self.train_test_crf(X_train, y_train, X_test, y_test, model_name)

    def lang_prediction(self, input_data, trained_model):

        if type(input_data) == list:
            X = [self.feature_extraction(input_data)]
            tokens = input_data
        else:
            tokens = sentence_splitter(input_data)
            X = [self.feature_extraction(tokens)]

        tag_prediction = trained_model.predict(X)
        doc_labels = tag_prediction[0]
        token_tag = [(token, tag) for token, tag in zip(tokens, doc_labels)]
        for t in token_tag:
            print(t[0] + ' ' + t[1])
