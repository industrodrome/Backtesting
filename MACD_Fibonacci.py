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

# Function to calculate MACD
def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculates MACD and Signal Line."""
    data['MACD'] = data['Close'].ewm(span=fast_period, adjust=False).mean() - data['Close'].ewm(span=slow_period, adjust=False).mean()
    data['MACD Signal'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()

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

# MACD + Fibonacci Retracement Strategy
def macd_fibonacci_strategy(data, slippage_factor=0.00, transaction_cost_percent=0.001, fill_probability=1):
    calculate_macd(data)  # Add MACD indicators
    fibonacci_levels = calculate_fibonacci_retracement(data)  # Get Fibonacci levels

    entry_points = []
    exit_points = []
    holding_position = False

    for i in range(1, len(data)):
        macd = data.loc[i, 'MACD']
        macd_signal = data.loc[i, 'MACD Signal']
        close_price = data.loc[i, 'Close']

        # Buy Signal: MACD crossover near Fibonacci levels
        if not holding_position and macd > macd_signal and data.loc[i - 1, 'MACD'] <= data.loc[i - 1, 'MACD Signal']:
            for level_name, level_price in fibonacci_levels.items():
                if abs(close_price - level_price) / close_price <= 0.02:  # Close to Fibonacci level
                    adjusted_price = apply_slippage(close_price, slippage_factor)
                    adjusted_price = apply_transaction_costs(adjusted_price, transaction_cost_percent)
                    adjusted_price = simulate_order_execution(adjusted_price, fill_probability)
                    entry_points.append((data.loc[i, 'Date'], adjusted_price))
                    holding_position = True
                    break

        # Sell Signal: MACD crossunder near Fibonacci levels
        elif holding_position and macd < macd_signal and data.loc[i - 1, 'MACD'] >= data.loc[i - 1, 'MACD Signal']:
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

# Get entry and exit points for MACD + Fibonacci strategy
macd_fibonacci_entry_points, macd_fibonacci_exit_points, fibonacci_levels = macd_fibonacci_strategy(data)
macd_fibonacci_entry_df = pd.DataFrame(macd_fibonacci_entry_points, columns=['Date', 'Buy Price'])
macd_fibonacci_exit_df = pd.DataFrame(macd_fibonacci_exit_points, columns=['Date', 'Sell Price'])
macd_fibonacci_trade_df = pd.DataFrame({
    'Buy Date': macd_fibonacci_entry_df['Date'],
    'Buy Price': macd_fibonacci_entry_df['Buy Price'],
    'Sell Date': macd_fibonacci_exit_df['Date'],
    'Sell Price': macd_fibonacci_exit_df['Sell Price']
})
macd_fibonacci_trade_df['Percentage Change'] = (macd_fibonacci_trade_df['Sell Price'] - macd_fibonacci_trade_df['Buy Price']) / macd_fibonacci_trade_df['Buy Price'] * 100
macd_fibonacci_total_percentage_increase = macd_fibonacci_trade_df['Percentage Change'].sum()

# Plot performance of MACD + Fibonacci strategy
def plot_macd_fibonacci_performance(data, strategy_trades, fibonacci_levels):
    plt.figure(figsize=(14, 8))
    plt.plot(data['Date'], data['Close'], label='Daily Close Price', linewidth=1)
    plt.plot(data['Date'], data['MACD'], label='MACD', linestyle='-', linewidth=1)
    plt.plot(data['Date'], data['MACD Signal'], label='MACD Signal', linestyle='--', linewidth=1)

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

    plt.title("MACD + Fibonacci Retracement Strategy Performance")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot performance for MACD + Fibonacci strategy
plot_macd_fibonacci_performance(data, macd_fibonacci_trade_df, fibonacci_levels)

# Print results
print("MACD + Fibonacci Retracement Strategy Total Percentage Increase: {:.2f}%".format(macd_fibonacci_total_percentage_increase))
