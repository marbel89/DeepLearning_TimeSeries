import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Model, Sequential

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

from tensorflow.keras.layers import Dense, Conv1D, LSTM, Lambda, Reshape, RNN, LSTMCell

import warnings

warnings.filterwarnings('ignore')


class DataWindow:
    """
        The DataWindow class manages data windowing for time series prediction based on the input width,
        label width, and shift.

        Attributes
        ----------
        train_df : pd.DataFrame
            DataFrame containing the training data.
        val_df : pd.DataFrame
            DataFrame containing the validation data.
        test_df : pd.DataFrame
            DataFrame containing the test data.
        label_columns : list of str
            List of columns to be used as labels.
        label_columns_indices : dict
            Dictionary mapping label column names to their indices.
        column_indices : dict
            Dictionary mapping column names to their indices.
        input_width : int
            Number of time steps used as input.
        label_width : int
            Number of time steps used as output.
        shift : int
            Offset between the end of inputs and the start of labels.
        total_window_size : int
            Total size of the window (input + shift).
        input_slice : slice
            Slice object for input data.
        input_indices : np.ndarray
            Array of indices for input data.
        label_start : int
            Starting index for label data.
        labels_slice : slice
            Slice object for label data.
        label_indices : np.ndarray
            Array of indices for label data.
        """

    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_to_inputs_labels(self, features):
        """
        Splits the features into inputs and labels based on the defined window.

        Parameters
        ----------
        features :
        The data to be split into inputs and labels.

        Returns
        -------
        tuple
        A tuple containing the inputs and labels.
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in
                 self.label_columns],
                axis=-1
            )
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='traffic_volume', max_subplots=3):
        """
        Plots the input, label, and prediction (if a model is provided) for the given data.

        Parameters
        ----------
        model : tf.keras.Model, optional
            The trained model used for making predictions. If None, only inputs and labels are plotted.
        plot_col : str, optional
            The column name to be plotted.
        max_subplots : int, optional
            The maximum number of subplots to display.
        """
        inputs, labels = self.sample_batch

        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel(f'{plot_col} [scaled]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col,
                                                                 None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', marker='s', label='Labels',
                        c='green', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='red', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time (h)')

    def make_dataset(self, data):
        """
        Creates a TensorFlow dataset from the provided data.

        Parameters
        ----------
        data : dataframe
            The data to be converted into a TensorFlow dataset.

        Returns
        -------
        tf.data.Dataset
            The created TensorFlow dataset.
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32
        )

        ds = ds.map(self.split_to_inputs_labels)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def sample_batch(self):
        result = getattr(self, '_sample_batch', None)
        if result is None:
            result = next(iter(self.train))
            self._sample_batch = result
        return result
