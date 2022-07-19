from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot.tf_keras import PlotLossesCallback
import os

def get_callbacks(root_path, model_name):
    joined_path = os.path.join(root_path, model_name)

    chkpt = ModelCheckpoint(joined_path, monitor='val_loss', verbose=1, save_best_only=True,
                            save_weights_only=False, mode='min')

    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=0, mode='max',
                                   baseline=None, restore_best_weights=False)

    callbacks = [PlotLossesCallback(), chkpt, early_stopping]

    return callbacks