import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from datawindow_class import DataWindow
from baseline import Baseline, MultiStepLastBaseline, RepeatBaseline

from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Lambda, Reshape, RNN, LSTMCell
import warnings

warnings.filterwarnings('ignore')
print(tf.__version__)

# Load datasets and health check them
train_df_ = pd.read_csv('train.csv', index_col=0)
val_df_ = pd.read_csv('val.csv', index_col=0)
test_df_ = pd.read_csv('test.csv', index_col=0)
print(train_df_.shape, val_df_.shape, test_df_.shape)

val_performance = {}
performance = {}

# Create DataWindow instances and pass data to each
single_step_window = DataWindow(input_width=1, label_width=1, shift=1, label_columns=['traffic_volume'],
                                train_df=train_df_, val_df=val_df_, test_df=test_df_)
wide_window = DataWindow(input_width=24, label_width=24, shift=1, label_columns=['traffic_volume'],
                         train_df=train_df_, val_df=val_df_, test_df=test_df_)
multi_window = DataWindow(input_width=24, label_width=24, shift=24, label_columns=['traffic_volume'],
                          train_df=train_df_, val_df=val_df_, test_df=test_df_)
mo_single_step_window = DataWindow(input_width=1, label_width=1, shift=1, label_columns=['temp', 'traffic_volume'],
                                   train_df=train_df_, val_df=val_df_, test_df=test_df_)
mo_wide_window = DataWindow(input_width=24, label_width=24, shift=1, label_columns=['temp', 'traffic_volume'],
                            train_df=train_df_, val_df=val_df_, test_df=test_df_)

# Create column index dictionary
column_indices = {name: i for i, name in enumerate(train_df_.columns)}


def eval_baselines():
    # Single Step Baseline
    baseline_last = Baseline(label_index=column_indices['traffic_volume'])
    baseline_last.compile(loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
    # Evaluate Single Step Baseline and save it to dict
    val_performance = {'Baseline - Last': baseline_last.evaluate(single_step_window.val)}
    performance = {'Baseline - Last': baseline_last.evaluate(single_step_window.test, verbose=0)}

    # Multi Step Baseline
    ms_baseline_last = MultiStepLastBaseline(label_index=column_indices['traffic_volume'])
    ms_baseline_last.compile(loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
    # Evaluate Multi Step Baseline and save it to dict
    ms_val_performance = {'Baseline - Last': ms_baseline_last.evaluate(multi_window.val)}
    ms_performance = {'Baseline - Last': ms_baseline_last.evaluate(multi_window.test, verbose=0)}

    # Multi Output Baseline: temp and traffic volume
    mo_baseline_last = Baseline(label_index=[column_indices['temp'], column_indices['traffic_volume']])
    mo_baseline_last.compile(loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
    # Evaluate Multi Output Baseline and save it to dict
    mo_val_performance = {'Baseline - Last': mo_baseline_last.evaluate(mo_wide_window.val)}
    mo_performance = {'Baseline - Last': mo_baseline_last.evaluate(mo_wide_window.test, verbose=0)}

    # 2nd baseline for multistep repeatedly
    ms_baseline_repeat = RepeatBaseline(label_index=column_indices['traffic_volume'])
    ms_baseline_repeat.compile(loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
    # Evaluate repeat ms
    ms_val_performance['Baseline - Repeat'] = ms_baseline_repeat.evaluate(multi_window.val)
    ms_performance['Baseline - Repeat'] = ms_baseline_repeat.evaluate(multi_window.test, verbose=0)

    # Print Performance Results in MAE
    print("Single Step Test MAE:", performance['Baseline - Last'][1])
    print("Multi Step Test MAE:", ms_performance['Baseline - Last'][1])
    print("Multi Output Test MAE:", mo_performance['Baseline - Last'][1])
    print("Multi Step Repeat MAE: ", ms_performance['Baseline - Repeat'][1])


check_eval_baselines = input("Evaluate baselines? (Y/N)")
if check_eval_baselines == "Y" or "y":
    eval_baselines()


# DL Model
def compile_and_fit(model, window, patience=3, max_epochs=50):
    """
    Compiles and fits the model using the provided data window.

    Parameters:
    model (keras.Model): The model to be compiled and trained.
    window (DataWindow): The data window containing train and validation datasets.
    patience (int): The number of epochs to wait for improvement before stopping.
    max_epochs (int): The maximum number of epochs to train the model.

    Returns:
    history: The history object containing training metrics.
    """

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   mode='min')

    model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(),
                  metrics=[MeanAbsoluteError()])

    history = model.fit(window.train,
                        epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])

    return history


# Models
linear_model = Sequential([
    Dense(units=1)
])

history = compile_and_fit(linear_model, single_step_window)
val_performance['Linear'] = linear_model.evaluate(single_step_window.val)
performance['Linear'] = linear_model.evaluate(single_step_window.test, verbose=0)


def plot_baseline(baseline_last=None, ms_baseline_last=None, mo_baseline_last=None, ms_baseline_repeat=None):
    # Plotting Results
    wide_window.plot(baseline_last)
    plt.show()
    multi_window.plot(ms_baseline_last)
    plt.show()
    plt.close()
    mo_wide_window.plot(mo_baseline_last)
    plt.show()
    mo_wide_window.plot(model=mo_baseline_last, plot_col='temp')
    plt.show()
    multi_window.plot(ms_baseline_repeat)
    plt.show()


def plot_dl_linear():
    wide_window.plot(linear_model)
    plt.show()


if check_eval_baselines == "Y" or "y":
    bl = input("Plot baseline? (Y/N)")
    if bl == "Y" or bl == "y":
        plot_baseline()
lin = input("Plot linear DL? (Y/N)")
if lin == "Y" or lin == "y":
    plot_dl_linear()
