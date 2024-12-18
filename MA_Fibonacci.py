import yfinance as yf
import pandas as pd
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

# Function to calculate Moving Averages
def calculate_moving_averages(data, short_window=20, long_window=50):
    """Calculates short and long moving averages."""
    data['Short MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long MA'] = data['Close'].rolling(window=long_window).mean()

# Function to calculate Fibonacci retracement levels
def calculate_fibonacci_retracement(data):
    """Calculates Fibonacci retracement levels."""
    highest_price = data['High'].max()
    lowest_price = data['Low'].min()
    diff = highest_price - lowest_price
    
    levels = {
        'Level 0%': highest_price,
        'Level 23.6%': highest_price - 0.236 * diff,
        'Level 38.2%': highest_price - 0.382 * diff,
        'Level 50%': highest_price - 0.5 * diff,
        'Level 61.8%': highest_price - 0.618 * diff,
        'Level 78.6%': highest_price - 0.786 * diff,
        'Level 100%': lowest_price,
    }
    return levels

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

# Moving Average + Fibonacci Retracement Strategy
def moving_average_fibonacci_strategy(data, short_window=20, long_window=50, slippage_factor=0.00, transaction_cost_percent=0.001, fill_probability=1):
    calculate_moving_averages(data, short_window, long_window)  # Add Moving Averages
    fibonacci_levels = calculate_fibonacci_retracement(data)  # Get Fibonacci levels

    entry_points = []
    exit_points = []
    holding_position = False

    for i in range(1, len(data)):
        short_ma = data.loc[i, 'Short MA']
        long_ma = data.loc[i, 'Long MA']
        close_price = data.loc[i, 'Close']

        # Buy Signal: Moving Average Golden Cross near Fibonacci levels
        if not holding_position and short_ma > long_ma and data.loc[i - 1, 'Short MA'] <= data.loc[i - 1, 'Long MA']:
            for level_name, level_price in fibonacci_levels.items():
                if abs(close_price - level_price) / close_price <= 0.02:  # Close to Fibonacci level
                    adjusted_price = apply_slippage(close_price, slippage_factor)
                    adjusted_price = apply_transaction_costs(adjusted_price, transaction_cost_percent)
                    adjusted_price = simulate_order_execution(adjusted_price, fill_probability)
                    entry_points.append((data.loc[i, 'Date'], adjusted_price))
                    holding_position = True
                    break

        # Sell Signal: Moving Average Death Cross near Fibonacci levels
        elif holding_position and short_ma < long_ma and data.loc[i - 1, 'Short MA'] >= data.loc[i - 1, 'Long MA']:
            for level_name, level_price in fibonacci_levels.items():
                if abs(close_price - level_price) / close_price <= 0.02:  # Close to Fibonacci level
                    adjusted_price = apply_slippage(close_price, slippage_factor)
                    adjusted_price = apply_transaction_costs(adjusted_price, transaction_cost_percent)
                    adjusted_price = simulate_order_execution(adjusted_price, fill_probability)
                    exit_points.append((data.loc[i, 'Date'], adjusted_price))
                    holding_position = False
                    break

    # Handle last holding position
    if holding_position:
        adjusted_price = apply_slippage(data.loc[len(data) - 1, 'Close'], slippage_factor)
        adjusted_price = apply_transaction_costs(adjusted_price, transaction_cost_percent)
        adjusted_price = simulate_order_execution(adjusted_price, fill_probability)
        exit_points.append((data.loc[len(data) - 1, 'Date'], adjusted_price))

    return entry_points, exit_points, fibonacci_levels

# Get entry and exit points for Moving Average + Fibonacci strategy
ma_fibonacci_entry_points, ma_fibonacci_exit_points, fibonacci_levels = moving_average_fibonacci_strategy(data)
ma_fibonacci_entry_df = pd.DataFrame(ma_fibonacci_entry_points, columns=['Date', 'Buy Price'])
ma_fibonacci_exit_df = pd.DataFrame(ma_fibonacci_exit_points, columns=['Date', 'Sell Price'])
ma_fibonacci_trade_df = pd.DataFrame({
    'Buy Date': ma_fibonacci_entry_df['Date'],
    'Buy Price': ma_fibonacci_entry_df['Buy Price'],
    'Sell Date': ma_fibonacci_exit_df['Date'],
    'Sell Price': ma_fibonacci_exit_df['Sell Price']
})
ma_fibonacci_trade_df['Percentage Change'] = (ma_fibonacci_trade_df['Sell Price'] - ma_fibonacci_trade_df['Buy Price']) / ma_fibonacci_trade_df['Buy Price'] * 100
ma_fibonacci_total_percentage_increase = ma_fibonacci_trade_df['Percentage Change'].sum()

# Plot performance of Moving Average + Fibonacci strategy
def plot_ma_fibonacci_performance(data, strategy_trades, fibonacci_levels):
    plt.figure(figsize=(14, 8))
    plt.plot(data['Date'], data['Close'], label='Daily Close Price', linewidth=1)
    plt.plot(data['Date'], data['Short MA'], label='Short Moving Average', linestyle='-', linewidth=1)
    plt.plot(data['Date'], data['Long MA'], label='Long Moving Average', linestyle='--', linewidth=1)

    # Plot Fibonacci levels
    for level_name, level_price in fibonacci_levels.items():
        plt.axhline(level_price, color='gray', linestyle='--', linewidth=0.8, label=f'{level_name}')

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

    plt.title("Moving Average + Fibonacci Retracement Strategy Performance")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot performance for Moving Average + Fibonacci strategy
plot_ma_fibonacci_performance(data, ma_fibonacci_trade_df, fibonacci_levels)

# Print results
print("Moving Average + Fibonacci Retracement Strategy Total Percentage Increase: {:.2f}%".format(ma_fibonacci_total_percentage_increase))
