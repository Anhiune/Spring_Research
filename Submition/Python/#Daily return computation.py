import pandas as pd

# Load your price data (already cleaned)
stock_path = r"C:\Users\hoang\Documents\Summer Research\python (locally download file)\Tesla Real-Life stock.csv"
df_stock = pd.read_csv(stock_path)

# Make sure date is in datetime format and sort ascending
df_stock['date'] = pd.to_datetime(df_stock['date'])
df_stock = df_stock.sort_values('date')

# Compute daily return
df_stock['return'] = df_stock['close_price'].pct_change(fill_method=None)

# Drop first row (return will be NaN)
df_stock = df_stock.dropna()

# Format date as string to merge later
df_stock['date'] = df_stock['date'].dt.strftime('%-m/%-d/%Y')

print(df_stock[['date', 'close_price', 'return']])