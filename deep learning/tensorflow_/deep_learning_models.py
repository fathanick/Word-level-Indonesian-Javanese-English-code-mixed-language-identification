import numpy as np
import tensorflow as tf
from crf.langid_crf import *
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Flatten
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional, Conv1D, MaxPool1D, Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.initializers import RandomUniform
from keras_crf import CRFModel
from livelossplot.tf_keras import PlotLossesCallback
np.random.seed(0)
plt.style.use("ggplot")

def blstm_model(num_words, num_tags, max_len):
    inputs = Input(shape=(max_len,))
    embd_layer = Embedding(input_dim=num_words, output_dim=50, input_length=max_len, mask_zero=True)(inputs)
    spa_dropout_layer = SpatialDropout1D(0.3)(embd_layer)
    blstm_layer = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.3))(spa_dropout_layer)
    out = TimeDistributed(Dense(num_tags, activation="softmax"))(blstm_layer)
    model = Model(inputs, out)

    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # plot_model(model, to_file="img_model/blstm.png")
    model.summary()

    return model

def blstm_crf_model(num_words, num_tags, max_len):
    # build backbone model, you can use large models like BERT
    inputs = Input(shape=(max_len,), dtype=tf.int32, name='inputs')
    embd_layer = Embedding(input_dim=num_words, output_dim=50, input_length=max_len, mask_zero=True)(inputs)
    spa_dropout_layer = SpatialDropout1D(0.3)(embd_layer)
    blstm_layer = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.3))(spa_dropout_layer)
    out = Dense(100, activation='relu')(blstm_layer)
    base = Model(inputs=inputs, outputs=out)

    model = CRFModel(base, num_tags)

    # no need to specify a loss for CRFModel, model will compute crf loss by itself
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        metrics=['acc']
    )
    model.summary()

    return model

def blstm_w2v_model(num_words, num_tags, max_len, embedding_weights):
    inputs = Input(shape=(max_len,))
    embd_layer = Embedding(input_dim=num_words, output_dim=50, input_length=max_len, weights=[embedding_weights],
                           trainable=False, mask_zero=True)(inputs)
    spa_dropout_layer = SpatialDropout1D(0.3)(embd_layer)
    blstm_layer = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.3))(spa_dropout_layer)
    out = TimeDistributed(Dense(num_tags, activation="softmax"))(blstm_layer)
    model = Model(inputs, out)

    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # plot_model(model, to_file="img_model/blstm.png")
    model.summary()

    return model


def wc_blstm_model(num_words, num_tags, num_char, max_len, max_len_char):
    # word embedding as input
    word_input = Input(shape=(max_len, ))
    word_embd = Embedding(input_dim=num_words + 2, output_dim=50, input_length=max_len, mask_zero=True)(word_input)

    # character embedding as input
    char_input = Input(shape=(max_len, max_len_char, ))
    char_embd = TimeDistributed(Embedding(input_dim=num_char + 2, output_dim = 10,
                                          input_length=max_len_char, mask_zero=True))(char_input)

    # character LSTM to obtain word encodings by characters
    char_encd = TimeDistributed(LSTM(units=20, return_sequences=False, recurrent_dropout=0.5))(char_embd)

    # main BLSTM stack
    concate = concatenate([word_embd, char_encd])
    spa_dropout_layer = SpatialDropout1D(0.3)(concate)
    blstm_layer = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.3))(spa_dropout_layer)
    out = TimeDistributed(Dense(num_tags + 1, activation='softmax'))(blstm_layer)
    model = Model([word_input, char_input], out)

    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    return model

def wc_cnn_blstm_model(num_words, num_tags, num_char, max_len, max_len_char):
    # word embedding as input
    word_input = Input(shape=(max_len, ))
    word_embd = Embedding(input_dim=num_words + 2, output_dim=50, input_length=max_len)(word_input)

    # character embedding as input
    char_input = Input(shape=(max_len, max_len_char, ))
    char_embd = TimeDistributed(Embedding(input_dim=num_char + 2, output_dim = 10,
                                          input_length=max_len_char,
                                          embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)))(char_input)

    dropout= Dropout(0.5)(char_embd)
    conv1d_out= TimeDistributed(Conv1D(kernel_size=3, filters=max_len_char, padding='same',activation='tanh', strides=1))(dropout)
    maxpool_out=TimeDistributed(MaxPooling1D(max_len_char))(conv1d_out)
    flat_layer = TimeDistributed(Flatten())(maxpool_out)
    char_encd = Dropout(0.5)(flat_layer)
    # character LSTM to obtain word encodings by characters
    #char_encd = TimeDistributed(LSTM(units=20, return_sequences=False, recurrent_dropout=0.5))(char_embd)

    # main BLSTM stack
    concate = concatenate([word_embd, char_encd])
    spa_dropout_layer = SpatialDropout1D(0.3)(concate)
    blstm_layer = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.3))(spa_dropout_layer)
    out = TimeDistributed(Dense(num_tags + 1, activation='softmax'))(blstm_layer)
    model = Model([word_input, char_input], out)

    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    return model

def wc_blstm_crf_model(num_words, num_tags, num_char, max_len, max_len_char):
    # word embedding as input
    word_input = Input(shape=(max_len, ))
    word_embd = Embedding(input_dim=num_words + 1, output_dim=50, input_length=max_len, mask_zero=True)(word_input)

    # character embedding as input
    char_input = Input(shape=(max_len, max_len_char, ))
    char_embd = TimeDistributed(Embedding(input_dim=num_char + 1, output_dim = 10,
                                          input_length=max_len_char, mask_zero=True))(char_input)

    # character LSTM to obtain word encodings by characters
    char_encd = TimeDistributed(LSTM(units=20, return_sequences=False, recurrent_dropout=0.5))(char_embd)

    # main BLSTM stack
    concate = concatenate([word_embd, char_encd])
    spa_dropout_layer = SpatialDropout1D(0.3)(concate)
    blstm_layer = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.3))(spa_dropout_layer)
    #out = Dense(100, activation='relu')(blstm_layer)
    out = TimeDistributed(Dense(num_tags, activation='softmax'))(blstm_layer)
    base = Model([word_input, char_input], out)

    model = CRFModel(base, num_tags)

    # no need to specify a loss for CRFModel, model will compute crf loss by itself
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        run_eagerly=True
    )
    model.summary()

    return model


def get_callbacks(root_path, model_name):
    joined_path = os.path.join(root_path, model_name)

    chkpt = ModelCheckpoint(joined_path, monitor='val_loss', verbose=1, save_best_only=True,
                            save_weights_only=False, mode='min')

    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=0, mode='max',
                                   baseline=None, restore_best_weights=False)

    callbacks = [PlotLossesCallback(), chkpt, early_stopping]

    return callbacks

def model_fitting(model, root_path, model_name, x_train, y_train, x_test, y_test, num_epoch, batch_sz):

    callbacks = get_callbacks(root_path, model_name)

    history = model.fit(
        x=x_train,
        y=y_train,
        # validation_data=(x_test, y_test),
        validation_split=0.1,
        epochs=num_epoch,
        callbacks=callbacks,
        verbose=1
    )

    print(model.evaluate(x_test, y_test))

    return history

def wc_embd_model_fitting(model, root_path, model_name, X_word_train, X_char_train, y_train, num_epoch, batch_sz,
                          max_len, max_len_char):

    callbacks = get_callbacks(root_path, model_name)

    history = model.fit([X_word_train,
                         np.array(X_char_train).reshape((len(X_char_train), max_len, max_len_char))],
                        np.array(y_train).reshape(len(y_train), max_len, 1),
                        validation_split=0.1,
                        batch_size=batch_sz,
                        epochs=num_epoch,
                        callbacks=callbacks,
                        verbose=1
                        )

    return history
