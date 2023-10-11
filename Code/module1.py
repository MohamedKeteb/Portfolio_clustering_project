import pandas as pd 
import yfinance as yf

def get_returns(start_date, end_date, ticker_list): 
    '''
    
    ----------------------------------------------------------------
    PURPOSE : create a dataframe of returns for a given period of time 
              in [start, end] for a list of tickers
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    PARAMS : 
    
    - start : datetime.datetime object
    
    - end : datetime.datetime object

    - ticker_list : list of tickers (strings)

    ----------------------------------------------------------------
    '''

    df = pd.DataFrame(yf.download(ticker_list, start_date, end_date))
    data = (df['close'] - df['open'])/df['open']
    data = data.dropna()
    return data