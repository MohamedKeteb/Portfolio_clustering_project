import pandas as pd 
import numpy as np 
import yfinance as yf
import plotly.express as px 
from sklearn.pipeline import Pipeline 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler 

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
    data = np.log(df['Close']/df['Open']).dropna()
    return df


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
   '''
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
   Y = pd.DataFrame(index=data.index)
   pipeline = Pipeline([
      ('scaler', StandardScaler()),   # Étape de standardisation
      ('kmeans', KMeans(n_clusters=5)) # Étape K-Means avec 3 clusters
      ])
   for i in range(n_repeat):
    pipeline.fit(data)
    predicted_labels = pipeline.named_steps['kmeans'].labels_
    data_with_clusters = pd.DataFrame(predicted_labels, index=data.index)
    y_i = "Clustering n°%i" % (i+1)
    Y[y_i] = data_with_clusters
    
   return Y


def cluster_composition(multiple_clustering):

    n_clustering = len(multiple_clustering.transpose()) - 1 ## minus 1 because we don't want to take into account the 
                                                            ## first column that corresponds to the tickers name

    names = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']

    Y = pd.DataFrame(index=names, columns=multiple_clustering.columns)

    for i in range(n_clustering): ## we range across the different clustering so as to recover the clustering composition at each step
        clustering = multiple_clustering.iloc[:, i]
        distinct_values = clustering.unique()

        for k, value in enumerate(distinct_values): 
            l = []
            for j in range(len(multiple_clustering.index)): 
                if multiple_clustering.iloc[j, i] == value:
                    l.append(multiple_clustering.index[j])

            Y.iloc[k, i] = l
    
    return Y

