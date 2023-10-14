import pandas as pd 
import yfinance as yf
import plotly.express as px 

def get_returns(start_date, end_date, ticker_list): 
    
    '''
    ----------------------------------------------------------------
    GENERAL IDEA : create a dataframe of returns for a given period of time 
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
    data = df['Adj Close'].pct_change()
    return data


def plot_stock_return(data): 

    
    '''
    ----------------------------------------------------------------
    GENERAL IDEA : plot the returns of different stock on the same 
                   time frame 
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    PARAMS : 
    
    - data : Pandas DataFrame object containing the returns of different
             stocks for a given period of time

    ----------------------------------------------------------------
    '''
    

    ticker_list = data.columns.to_list()
    fig = px.line(data, data.index, [data[ticker] for ticker in ticker_list], title='Stock returns')
    fig.update_layout(title="Stock returns", legend_title="Stocks")
    fig.update_yaxes(title_text="Returns")
    fig.show()



def multiple_clusterings(n_repeat, data, model):
     Y = pd.DataFrame(index=data.index)
     for i in range(n_repeat):
        model.fit(data)
        predicted_labels = model.predict(data)
        data_with_clusters = pd.DataFrame(predicted_labels, index=data.index)
        y_i = "Clustering n°%i" % (i+1)
        Y[y_i] = data_with_clusters
     return Y 
   

    
'''
   ----------------------------------------------------------------
    ## Type of data ##

    n_repeat : integer --> number of time we apply the clustering method
    model : sklearn model we use --> e.g. GaussianMixture()
    data : pd.DataFrame --> data we want to fit to the model

    -------------

    ## Output ##

    Outputs a pandas DataFrame object of shape (len(data.index), n_repeat)

    -------------

    ## Genera idea ##

    The idea is here to train the model on the dataset data multiple time (here n_repeat time)
    and create a DataFrame whose columns are the cluster labels of each stock and whose rows are
    the label of a given stock for each clustering method

    '''