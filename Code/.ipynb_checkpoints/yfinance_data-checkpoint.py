import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from bs4 import BeautifulSoup
import requests 
import csv 


with open('S&P500_symbols.csv', mode='r') as file:
    # Create a CSV reader
    csv_reader = csv.reader(file)
    
    # Transform the CSV data into a list
    stock_symbols = []
    for row in csv_reader:
        stock_symbols.append(row[0])

# Close the file
file.close()
stock_symbols.pop(0)



start = "2022-01-01" # start date
end = "2022-12-01" # end date

df = pd.DataFrame(yf.download(stock_symbols, start, end)) # data on the 198 assets
data = np.log(df["Close"]/df["Open"]).transpose() # compute the returns of these assets
data = data.dropna()

