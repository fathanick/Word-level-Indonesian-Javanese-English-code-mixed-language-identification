import random
import scipy.stats
import sklearn_crfsuite
import pickle
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
import eli5
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, make_scorer
from sklearn_crfsuite import metrics
from helper.splitter import sentence_splitter
from helper.data_transformer import to_token_tag_list
from warnings import simplefilter  # import warnings filter
from collections import Counter

simplefilter(action='ignore', category=FutureWarning)  # ignore all future warnings


class LanguageIdentifier:
    model = None
    window = 5
    symbols = ['@', '#', '&', '$']
    labels = ['ID', 'JV', 'EN', 'MIX-ID-EN', 'MIX-ID-JV', 'MIX-JV-EN', 'O']
    c1 = 0.1
    c2 = 0.1

    def __init__(self):
        random.seed(0)
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',  # for gradient descent for optimization and getting model parameters
            c1=0.016119422903376198,  # Coefficient for Lasso (L1) regularization for
            c2=0.05918362699721527,  # Coefficient for Ridge (L2) regularization
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
        # apply features for each token

        token = sentence[index][0]

        features = {
            'token.lower': token.lower(),
            'n_gram_0': token, # only show 5 characters, more than 5 chars will be cut
            'token_BOS': index == 0,
            'token_EOS': index == len(sentence) - 1,
            'prev_tag': '' if index == 0 else sentence[index - 1][1],
            'next_tag': '' if index == len(sentence) - 1 else sentence[index + 1][1],
            'prev_2tag': '' if index == 0 or index == 1 else sentence[index - 2][1],
            'next_2tag': '' if index == len(sentence) - 1 or index == len(sentence) - 2 else sentence[index + 2][1],
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
        data = to_token_tag_list(data)

        X = [self.sent2features(s) for s in data]
        y = [self.sent2tags(s) for s in data]

        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
        }

        f1_scorer = make_scorer(metrics.flat_f1_score,
                                average='weighted',
                                labels=self.labels)

        rs = RandomizedSearchCV(self.model,
                                params_space,  # pass the dictionary of parameters that we need to optimize
                                cv=10,  # Determines the cross-validation splitting strategy
                                verbose=1,  # Controls the verbosity: the higher, the more messages
                                n_jobs=-1,  # Number of jobs to run in parallel, -1 means using all processors
                                n_iter=50,  # Number of parameter settings that are sampled
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
        print(classification_report(flat_y, flat_y_pred, labels=self.labels, digits=4))

        # sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        fig, ax = plt.subplots(figsize=(10, 8))

        # plt.figure(figsize=(12, 10))
        # sns.set(rc={'figure.figsize': (14, 12)})
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, annot_kws={'size': 16})
        # annot=True to annotate cells, ftm='g' to disable scientific notation

        # labels, title and ticks
        ax.set_title('Confusion Matrix', fontsize=16)
        ax.set_xlabel('Predicted labels', fontsize=16)
        ax.set_ylabel('True labels', fontsize=16)
        ax.xaxis.set_ticklabels(self.labels, rotation=45, fontsize=16)
        ax.yaxis.set_ticklabels(self.labels, rotation=0, fontsize=16)

        plt.show()

    def save_model(self, model_name):
        # save the model to disk
        root_path = 'model/'
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

    def train_test_crf(self, X_train, y_train, X_test):
        # train the data
        crf_model = self.model.fit(X_train, y_train)

        # prediction
        y_pred_test = self.model.predict(X_test)
        y_pred_train = self.model.predict(X_train)

        return crf_model, y_pred_train, y_pred_test

    def result_performance(self, y_train, y_test, y_pred_train, y_pred_test, model_name):
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

        # eli5.show_weights(self.model)
        # save model
        self.save_model(model_name)

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
