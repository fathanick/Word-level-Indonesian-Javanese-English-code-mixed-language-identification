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