import pandas as pd
import yfinance as yf



all_tickers = yf.Tickers('')


symbols_list = all_tickers.tickers


symbols = [ticker.info['symbol'] for ticker in symbols_list]


print(all_tickers)
