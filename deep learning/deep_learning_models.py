# %matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
plt.style.use("ggplot")
import tensorflow as tf
from crf.langid_crf import *
from helper.dataset_reader import read_tsv
from helper.data_transformer import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Flatten
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot.tf_keras import PlotLossesCallback


def get_unique_words(df):
    unique_words = list(set(df['Token'].values))
    unique_words.append('ENDPAD')

    return unique_words

def get_unique_tags(df):
    tags = list(set(df['Label'].values))

    return tags

def plot_stc_length(dt_pair):
    plt.hist([len(s) for s in dt_pair], bins=50)
    plt.show()

def word2idx(words):
    return {w: i + 1 for i, w in enumerate(words)}

def tag2idx(tags):
    return {t: i for i, t in enumerate(tags)}

def input_data(words, tags, dt_pair):
    max_len = 50
    num_words = len(words)

    # word2idx = {w: i + 1 for i, w in enumerate(words)}
    # tag2idx = {t: i for i, t in enumerate(tags)}
    word_idx = word2idx(words)
    tag_idx = tag2idx(tags)

    X = [[word_idx[w[0]] for w in s] for s in dt_pair]
    X = pad_sequences(maxlen=max_len, sequences=X, padding='post', value=num_words-1)

    y = [[tag_idx[t[1]] for t in s] for s in dt_pair]
    y = pad_sequences(maxlen=max_len, sequences=y, padding='post', value=tag_idx["O"])

    return X, y

def blstm_model(num_words, num_tags, max_len):
    model = Sequential()
    model.add(Embedding(input_dim=num_words, output_dim=50, input_length=max_len))
    model.add(SpatialDropout1D(0.1))
    model.add(Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1)))
    model.add(TimeDistributed(Dense(num_tags, activation="softmax")))
    model.summary()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

def training_history(model, x_train, y_train, x_test, y_test, num_epoch, batch_sz):
    chkpt = ModelCheckpoint("model/model_weights.h5", monitor='val_loss', verbose=1, save_best_only=True,
                            save_weights_only=True, mode='min')

    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=2, verbose=0, mode='max',
                                   baseline=None, restore_best_weights=False)

    callbacks = [PlotLossesCallback(), chkpt, early_stopping]

    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_sz,
        epochs=num_epoch,
        callbacks=callbacks,
        verbose=1
    )

    print(model.evaluate(x_test, y_test))

    return history

def pred_idx2tag(y_pred, tags):
    predicted_label = []
    tag_idx = tag2idx(tags)

    idx2tag = {i: w for w, i in tag_idx.items()}

    for pred_i in y_pred:
        for p in pred_i:
            p_i = np.argmax(p)
            predicted_label.append(idx2tag[p_i].replace("ENDPAD", "O"))

    return predicted_label

def actual_idx2tag(y_test, tags):
    actual_label = []
    tag_idx = tag2idx(tags)

    idx2tag = {i: w for w, i in tag_idx.items()}

    for act_i in y_test:
        for a in act_i:
            actual_label.append(idx2tag[a])

    return actual_label

def performance_report(model, x_test, y_test, tags, df):
    y_actual = actual_idx2tag(y_test=y_test,tags=tags)
    y_pred = model.predict(x_test, verbose=1)
    y_pred = pred_idx2tag(y_pred, tags)
    labels = ['ID', 'JV', 'EN', 'MIX-ID-EN', 'MIX-ID-JV', 'MIX-JV-EN', 'O']

    print(classification_report(y_actual, y_pred, labels=labels))

    cm = confusion_matrix(y_actual, y_pred)
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
    ax.xaxis.set_ticklabels(labels, rotation=45, fontsize=16)
    ax.yaxis.set_ticklabels(labels, rotation=0, fontsize=16)

    plt.show()

if __name__ == "__main__":
    data = read_tsv('../dataset/comlid-data-140422-v1.tsv')
    all_data, words, tags = data
    df = list_to_dataframe(data)
    words = get_unique_words(df)
    tags = get_unique_tags(df)

    dt_pair = to_token_tag_list(data)
    X, y = input_data(words, tags, dt_pair)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    num_words = len(words)
    num_tags = len(tags)
    max_len = 50
    model = blstm_model(num_words, num_tags, max_len)

    training_history(model, x_train, y_train, x_test, y_test, num_epoch=20, batch_sz=32)
    performance_report(model, x_test, y_test, tags, df)

