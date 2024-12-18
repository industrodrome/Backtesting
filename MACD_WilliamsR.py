import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Fetch daily data for TATACONSUM.NS from yfinance
ticker = 'TATACONSUM.NS'
start_date = '2022-01-01'
end_date = '2024-08-01'

# Fetch the data
data = yf.download(ticker, start=start_date, end=end_date)

# Prepare the data
data['Date'] = data.index
data.reset_index(drop=True, inplace=True)

# Function to calculate MACD
def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculates MACD and Signal Line."""
    data['MACD'] = data['Close'].ewm(span=fast_period, adjust=False).mean() - data['Close'].ewm(span=slow_period, adjust=False).mean()
    data['MACD Signal'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()

# Function to calculate William's %R
def calculate_williams_r(data, lookback_period=14):
    """Calculates William's %R."""
    highest_high = data['High'].rolling(window=lookback_period).max()
    lowest_low = data['Low'].rolling(window=lookback_period).min()
    data['Williams %R'] = -100 * ((highest_high - data['Close']) / (highest_high - lowest_low))

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

# MACD + William's %R Strategy
def macd_williams_strategy(data, slippage_factor=0.00, transaction_cost_percent=0.001, fill_probability=1):
    calculate_macd(data)  # Add MACD indicators
    calculate_williams_r(data)  # Add William's %R

    entry_points = []
    exit_points = []
    holding_position = False

    for i in range(1, len(data)):
        macd = data.loc[i, 'MACD']
        macd_signal = data.loc[i, 'MACD Signal']
        williams_r = data.loc[i, 'Williams %R']
        close_price = data.loc[i, 'Close']

        # Buy Signal: MACD crossover + William's %R oversold
        if not holding_position and macd > macd_signal and williams_r < -80:
            adjusted_price = apply_slippage(close_price, slippage_factor)
            adjusted_price = apply_transaction_costs(adjusted_price, transaction_cost_percent)
            adjusted_price = simulate_order_execution(adjusted_price, fill_probability)
            entry_points.append((data.loc[i, 'Date'], adjusted_price))
            holding_position = True

        # Sell Signal: MACD crossunder + William's %R overbought
        elif holding_position and macd < macd_signal and williams_r > -20:
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

# Get entry and exit points for MACD + William's %R strategy
macd_williams_entry_points, macd_williams_exit_points = macd_williams_strategy(data)
macd_williams_entry_df = pd.DataFrame(macd_williams_entry_points, columns=['Date', 'Buy Price'])
macd_williams_exit_df = pd.DataFrame(macd_williams_exit_points, columns=['Date', 'Sell Price'])
macd_williams_trade_df = pd.DataFrame({
    'Buy Date': macd_williams_entry_df['Date'],
    'Buy Price': macd_williams_entry_df['Buy Price'],
    'Sell Date': macd_williams_exit_df['Date'],
    'Sell Price': macd_williams_exit_df['Sell Price']
})
macd_williams_trade_df['Percentage Change'] = (macd_williams_trade_df['Sell Price'] - macd_williams_trade_df['Buy Price']) / macd_williams_trade_df['Buy Price'] * 100
macd_williams_total_percentage_increase = macd_williams_trade_df['Percentage Change'].sum()

# Plot performance of MACD + William's %R strategy
def plot_macd_williams_performance(data, strategy_trades):
    plt.figure(figsize=(14, 8))
    plt.plot(data['Date'], data['Close'], label='Daily Close Price', linewidth=1)
    plt.plot(data['Date'], data['MACD'], label='MACD', linestyle='-', linewidth=1)
    plt.plot(data['Date'], data['MACD Signal'], label='MACD Signal', linestyle='--', linewidth=1)

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

    plt.title("MACD + William's %R Strategy Performance")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot performance for MACD + William's %R strategy
plot_macd_williams_performance(data, macd_williams_trade_df)

# Print results
print("MACD + William's %R Strategy Total Percentage Increase: {:.2f}%".format(macd_williams_total_percentage_increase))
