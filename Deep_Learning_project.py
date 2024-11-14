# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import yfinance as yf
from datetime import datetime


# Define a function to fetch stock data
def fetch_stock_data(tickers, start, end):
    try:
        stock_data = {}
        for ticker in tickers:
            stock_data[ticker] = yf.download(ticker, start=start, end=end)
        return stock_data
    except Exception as e:
        print(f"Error fetching stock data: {e}")


# Define a function to clean the data
def clean_data(stock_data):
    for ticker in stock_data.keys():
        stock_data[ticker].dropna(inplace=True)
    return stock_data


# Define a function to visualize the data
def visualize_data(stock_data):
    plot_closing_prices(stock_data)
    plot_moving_averages(stock_data)
    plot_daily_returns(stock_data)


# Define a function to plot closing prices
def plot_closing_prices(stock_data):
    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    for i, ticker in enumerate(stock_data.keys(), 1):
        plt.subplot(2, 2, i)
        stock_data[ticker]['Adj Close'].plot()
        plt.ylabel('Adj Close')
        plt.title(f"Closing Price of {ticker}")

    plt.tight_layout()


# Define a function to plot moving averages
def plot_moving_averages(stock_data, ma_days=[10, 20, 50]):
    for ticker in stock_data.keys():
        for ma in ma_days:
            stock_data[ticker][f"MA {ma} days"] = stock_data[ticker]['Adj Close'].rolling(ma).mean()

    fig, axes = plt.subplots(2, 2, figsize=(10, 3))
    fig.tight_layout()

    for i, (ticker, data) in enumerate(stock_data.items()):
        ax = axes[i // 2, i % 2]
        data[['Adj Close', 'MA 10 days', 'MA 20 days', 'MA 50 days']].plot(ax=ax)
        ax.set_title(ticker)


# Define a function to plot daily returns
def plot_daily_returns(stock_data):
    plt.figure(figsize=(12, 9))
    for i, (ticker, data) in enumerate(stock_data.items(), 1):
        data['Daily Return'] = data['Adj Close'].pct_change()
        plt.subplot(2, 2, i)
        data['Daily Return'].hist(bins=50)
        plt.title(ticker)

    plt.tight_layout()


# Define the main function
def main():
    # Tech stocks
    tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
    company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]

    # Date range
    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)

    # Fetch stock data
    stock_data = fetch_stock_data(tech_list, start, end)

    # Clean the data
    stock_data = clean_data(stock_data)

    # Add company name column for each DataFrame
    for ticker, name in zip(tech_list, company_name):
        stock_data[ticker]["Company"] = name

    # Concatenate all stock data
    df = pd.concat(stock_data.values(), axis=0)

    # Visualize the data
    visualize_data(stock_data)

    # Get the stock quote
    df = yf.download('AAPL', start='2012-01-01', end=datetime.now())

    # Show the data
    print(df)

    # Plot the closing price history
    plt.figure(figsize=(16, 6))
    plt.title('Close Price History')
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.show()

    # Create a new dataframe with only the 'Close column
    data = df.filter(['Close'])

    # Convert the dataframe to a numpy array
    dataset = data.values

    # Get the number of rows to train the model on
    training_data_len = int(np.ceil(len(dataset) * 0.95))

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set
    x_train = []
    y_train = []

    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=10, epochs=15)

    # Create the testing data set
    test_data = scaled_data[training_data_len - 60:, :]

    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

    # Print the RMSE
    print(f"RMSE: {rmse}")

    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    # Visualize the data
    plt.figure(figsize=(16, 6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

    # Show the valid and predicted prices
    print(valid)


if __name__ == "__main__":
    main()