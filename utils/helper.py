import yfinance as yf
import pandas as pd

def load_data(tickers, start="2020-01-01", end="2025-01-01"):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data.dropna()

def calculate_returns(prices):
    return prices.pct_change().dropna()
