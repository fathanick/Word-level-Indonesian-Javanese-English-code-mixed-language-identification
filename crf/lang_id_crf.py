import random
import sklearn_crfsuite
import pickle
import os
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics
from sklearn.metrics import confusion_matrix
from warnings import simplefilter # import warnings filter
simplefilter(action='ignore', category=FutureWarning) # ignore all future warnings

class LanguageIdentifier:

    model = None
    window = 5
    #labels = ['ID', 'JV', 'EN', 'O', 'MIX-ID-EN', 'MIX-ID-JV', 'MIX-JV-EN']
    labels = ['ID', 'JV', 'EN', 'NE', 'O', 'MIX-ID-EN', 'MIX-ID-JV', 'MIX-JV-EN']

    def __init__(self):
        random.seed(0)
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',  # for gradient descent for optimization and getting model parameters
            c1=0.1,  # Coefficient for Lasso (L1) regularization
            c2=0.1,  # Coefficient for Ridge (L2) regularization
            max_iterations=100, # The maximum number of iterations for optimization algorithms, iteration for the gradient descent optimization
            all_possible_transitions=True
            # Specify whether CRFsuite generates transition features that do not even occur in the training data
        )

    '''
    def feature_extraction_basic(self, tokens):
        features = []

        for idx, token in enumerate(tokens):
            feature = {
                'n_gram_0': token,
                'is_alpha': token.isalpha(),
                'is_numeric': token.isnumeric(),
                'is_capital': token.isupper(),
                'contains_alpha': bool(re.search('[a-zA-Z]', token)),
                'contains_numeric': bool(re.search('[0-9]', token)),
                'contains_aphostrope': '\'' in token,
            }

            if len(token) > 5:
                for i in range(1, len(token) - self.window):
                    feature[f'n_gram_{i}'] = token[i:(i + self.window)]

            features.append(feature)

        return features
    '''
    def feature_extraction(self, tokens):
        features = []

        for idx, token in enumerate(tokens):
            feature = {
                'n_gram_0': token,
                'is_alpha': token.isalpha(),
                'is_numeric': token.isnumeric(),
                'is_capital': token.isupper(),
                'is_punctuation': (token in string.punctuation),
                'contains_alpha': bool(re.search('[a-zA-Z]', token)),
                'contains_numeric': bool(re.search('[0-9]', token)),
                'contains_aphostrope': '\'' in token,
            }

            if len(token) > 5:
                for i in range(1, len(token) - self.window):
                    feature[f'n_gram_{i}'] = token[i:(i + self.window)]

            features.append(feature)

        return features

    def data_transformer(self, data):
        x = []
        y = []

        for tokens, tags in data[0]:
            x.append(tokens)
            y.append(tags)

        return x, y

    '''
    X = list of tokens.
    y = list of tags for each token. 
    '''

    def train(self, X, y):
        X_train = []

        for tokens in X:
            X_train.append(self.feature_extraction(tokens))

        self.model.fit(X_train, y)

    def predict(self, X):
        features = []

        for tokens in X:
            features.append(self.feature_extraction(tokens))

        y_pred = self.model.predict(features)

        return y_pred

    def show_confusion_matrix(self, y, y_pred):
        # show classification report
        print(metrics.flat_classification_report(y, y_pred, labels=self.labels))

        # create a confusion matrix
        print('Confusion Matrix')
        flat_y = [item for y_ in y for item in y_]
        flat_y_pred = [item for y_pred_ in y_pred for item in y_pred_]
        cm = confusion_matrix(flat_y, flat_y_pred, labels=self.labels)
        # print(cm)

        # sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax = plt.subplot()
        sns.set(rc={'figure.figsize': (20, 18)})
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)  # annot=True to annotate cells, ftm='g' to disable scientific notation

        # labels, title and ticks
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.xaxis.set_ticklabels(self.labels, rotation=90)
        ax.yaxis.set_ticklabels(self.labels, rotation=0)

        #root_path = '../images/'
        #joined_path = os.path.join(root_path, )

        #plt.savefig()
        plt.show()

    def save_model(self, model_name):
        # save the model to disk
        root_path = '../model/'
        joined_path = os.path.join(root_path, model_name)
        pickle.dump(self.model, open(joined_path, 'wb'))

    def pipeline(self, data, test_size, model_name):
        # transform data
        X, y = self.data_transformer(data)

        # split data into train and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        # train the data
        self.train(X_train, y_train)

        # prediction
        y_pred_test = self.predict(X_test)
        y_pred_train = self.predict(X_train)

        # show confusion matrix
        print('\n Evaluation on the test data')
        self.show_confusion_matrix(y_test, y_pred_test)
        print('\n Evaluation on the training data')
        self.show_confusion_matrix(y_train, y_pred_train)

        # save model
        self.save_model(model_name)

