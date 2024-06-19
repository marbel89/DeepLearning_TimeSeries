# Deep Learning Time Series Prediction

This project focuses on time series prediction using deep learning models. The repository contains the implementation of data windowing, model baselines, and the main script for training, plotting, and evaluating the models.

## Project Structure

- `baseline.py`: Defines baseline models for comparison.
- `datawindow_class.py`: Manages data windowing for time series prediction.
- `main.py`: Main script for training the model, plotting results, and evaluating performance.

## Files and Modules

### `baseline.py`
This file contains baseline models that serve as reference points for evaluating the performance of more complex models.

- **Baseline**: A simple model that either returns the input or selects specific labels from the inputs.
- **MultiStepLastBaseline**: A model that replicates the last input value for multiple steps.
- **RepeatBaseline**: A model that repeats a specific label index across multiple steps.

### `datawindow_class.py`
This file manages data windowing for time series prediction. It prepares the data for training, validation, and testing by creating sliding windows.

- **DataWindow Class**: Handles the creation of input and label windows based on specified widths and shifts. It includes methods for splitting data, creating TensorFlow datasets, and plotting results.

### `main.py`
This is the main script for the project. It loads the data, creates instances of `DataWindow`, trains the models, and evaluates their performance.

- **Data Loading**: Loads the training, validation, and test datasets.
- **Data Windowing**: Creates data windows for the models.
- **Model Training**: Trains baseline models and evaluates their performance.
- **Plotting**: Plots the input data, labels, and model predictions using subplots for comparison.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Seaborn

### Usage

  - Prepare your data: Run the Jupyer Notebook "dl_feature_engineering_scaling"
  - Ensure you have the training, validation, and test datasets (train.csv, val.csv, test.csv) in the project directory.

    Run the main script:
    ```bash
    python main.py
    ```

    This script will load the data, create data windows, train the models, evaluate their performance, and plot the results.

