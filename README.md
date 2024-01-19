# Stock-price-prediction-using-LSTM
This is a stock price prediction project created using Python.

# Recurrent Neural Network (RNN) Project

This README file provides an overview of the Recurrent Neural Network (RNN) project(LSTM). In this project, we will build an RNN model for predicting Google's stock price based on historical stock price data. The project is organized into several sections, including data preprocessing, building the RNN architecture, training the model, and making predictions.

## Project Overview

The objective of this project is to create an RNN model capable of predicting Google's stock price. The project consists of the following steps:

1. **Data Preprocessing**: We perform data preprocessing to prepare the dataset for training. This includes loading historical stock price data, scaling the data, and creating a data structure with appropriate time steps.

2. **Building the RNN**: We design the architecture of the Recurrent Neural Network (RNN) model, specifying the LSTM layers, dropout layers, and output layer.

3. **Training the RNN**: The RNN model is compiled with an optimizer and loss function, and it is then trained on the training data using the `fit` method.

4. **Making Predictions**: After training, we demonstrate how to make predictions on future stock prices using the trained model. We also visualize the predicted stock prices alongside the real stock prices.

## Getting Started

Before running the code, ensure that you have the required libraries installed, including `numpy`, `pandas`, `matplotlib`, and `keras`. You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib keras
```

## Usage

1. **Data Preprocessing**: The code starts with data preprocessing steps, including loading historical stock price data and scaling it using Min-Max scaling.

2. **Building the RNN**: We define the RNN architecture, including LSTM layers with dropout to prevent overfitting.

3. **Training the RNN**: The RNN model is compiled with an optimizer and loss function. It is then trained on the training data with a specified number of epochs and batch size.

4. **Making Predictions**: The code demonstrates how to make predictions on future stock prices using the trained model. It loads test data, preprocesses it, and predicts stock prices. Real and predicted stock prices are visualized using matplotlib.

## Example Predictions

To make predictions on future stock prices, you can use the following code snippet as an example:

```python
# Load the test data
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Get the Predicted stock price of 2017
# (code for getting test data and predictions is provided in the project code)

# Visualize the results
# (code for plotting real and predicted stock prices is provided in the project code)
```

## Conclusion

This RNN project demonstrates how to build, train, and use a Recurrent Neural Network model for time series prediction tasks, specifically for predicting stock prices. You can further customize the model architecture, fine-tune hyperparameters, and apply it to other time series forecasting projects.
