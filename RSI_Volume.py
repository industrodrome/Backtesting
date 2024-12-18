import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import random

# Fetch daily data for TATACONSUM.NS from yfinance
ticker = 'TATASTEEL.NS'
start_date = '2022-01-01'
end_date = '2024-08-01'

# Fetch the data
data = yf.download(ticker, start=start_date, end=end_date)

# Prepare the data
data['Date'] = data.index
data.reset_index(drop=True, inplace=True)

# Function to calculate RSI
def calculate_rsi(data, period=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

# Function to calculate Volume Oscillator
def calculate_volume_oscillator(data, short_window=14, long_window=28):
    """Calculates the Volume Oscillator."""
    data['Short Volume MA'] = data['Volume'].rolling(window=short_window).mean()
    data['Long Volume MA'] = data['Volume'].rolling(window=long_window).mean()
    data['Volume Oscillator'] = ((data['Short Volume MA'] - data['Long Volume MA']) / data['Long Volume MA']) * 100

# Function to simulate slippage
def apply_slippage(price, slippage_factor=0.001):
    """Applies slippage to the given price."""
    slippage = price * slippage_factor * (random.choice([-1, 1]))
    return price + slippage

# Function to calculate transaction costs
def apply_transaction_costs(price, cost_percent=0.001):
    """Applies transaction costs to the given price."""
    transaction_cost = price * cost_percent
    return price - transaction_cost

# Function to simulate partial order execution
def simulate_order_execution(price, fill_probability=0.95):
    """Simulates order execution with the possibility of partial fill."""
    if random.random() > fill_probability:
        return price * random.uniform(0.5, 0.9)  # Partial fill
    return price

# RSI + Volume Oscillator Strategy
def rsi_volume_strategy(data, rsi_lower=30, rsi_upper=70, slippage_factor=0.00, transaction_cost_percent=0.001, fill_probability=1):
    calculate_rsi(data)  # Add RSI
    calculate_volume_oscillator(data)  # Add Volume Oscillator

    entry_points = []
    exit_points = []
    holding_position = False

    for i in range(1, len(data)):
        rsi = data.loc[i, 'RSI']
        volume_oscillator = data.loc[i, 'Volume Oscillator']
        close_price = data.loc[i, 'Close']

        # Buy Signal: RSI crosses above lower threshold, Volume Oscillator positive
        if not holding_position and rsi > rsi_lower and data.loc[i - 1, 'RSI'] <= rsi_lower and volume_oscillator > 0:
            adjusted_price = apply_slippage(close_price, slippage_factor)
            adjusted_price = apply_transaction_costs(adjusted_price, transaction_cost_percent)
            adjusted_price = simulate_order_execution(adjusted_price, fill_probability)
            entry_points.append((data.loc[i, 'Date'], adjusted_price))
            holding_position = True

        # Sell Signal: RSI crosses below upper threshold, Volume Oscillator negative
        elif holding_position and rsi < rsi_upper and data.loc[i - 1, 'RSI'] >= rsi_upper and volume_oscillator < 0:
            adjusted_price = apply_slippage(close_price, slippage_factor)
            adjusted_price = apply_transaction_costs(adjusted_price, transaction_cost_percent)
            adjusted_price = simulate_order_execution(adjusted_price, fill_probability)
            exit_points.append((data.loc[i, 'Date'], adjusted_price))
            holding_position = False

    # Handle last holding position
    if holding_position:
        adjusted_price = apply_slippage(data.loc[len(data) - 1, 'Close'], slippage_factor)
        adjusted_price = apply_transaction_costs(adjusted_price, transaction_cost_percent)
        adjusted_price = simulate_order_execution(adjusted_price, fill_probability)
        exit_points.append((data.loc[len(data) - 1, 'Date'], adjusted_price))

    return entry_points, exit_points

# Get entry and exit points for RSI + Volume Oscillator strategy
rsi_volume_entry_points, rsi_volume_exit_points = rsi_volume_strategy(data)
rsi_volume_entry_df = pd.DataFrame(rsi_volume_entry_points, columns=['Date', 'Buy Price'])
rsi_volume_exit_df = pd.DataFrame(rsi_volume_exit_points, columns=['Date', 'Sell Price'])
rsi_volume_trade_df = pd.DataFrame({
    'Buy Date': rsi_volume_entry_df['Date'],
    'Buy Price': rsi_volume_entry_df['Buy Price'],
    'Sell Date': rsi_volume_exit_df['Date'],
    'Sell Price': rsi_volume_exit_df['Sell Price']
})
rsi_volume_trade_df['Percentage Change'] = (rsi_volume_trade_df['Sell Price'] - rsi_volume_trade_df['Buy Price']) / rsi_volume_trade_df['Buy Price'] * 100
rsi_volume_total_percentage_increase = rsi_volume_trade_df['Percentage Change'].sum()

# Plot performance of RSI + Volume Oscillator strategy
def plot_rsi_volume_performance(data, strategy_trades):
    plt.figure(figsize=(14, 8))
    plt.plot(data['Date'], data['Close'], label='Daily Close Price', linewidth=1)

    # Highlight buy and sell signals
    for index, row in strategy_trades.iterrows():
        buy_date = row['Buy Date']
        buy_price = row['Buy Price']
        sell_date = row['Sell Date']
        sell_price = row['Sell Price']

        # Plot buy signals
        plt.plot(buy_date, buy_price, marker='^', color='green', markersize=10, label='Buy Signal' if index == 0 else "")

        # Plot sell signals
        plt.plot(sell_date, sell_price, marker='v', color='red', markersize=10, label='Sell Signal' if index == 0 else "")

    plt.title("RSI + Volume Oscillator Strategy Performance")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot performance for RSI + Volume Oscillator strategy
plot_rsi_volume_performance(data, rsi_volume_trade_df)

# Print results
print("RSI + Volume Oscillator Strategy Total Percentage Increase: {:.2f}%".format(rsi_volume_total_percentage_increase))
