import pandas as pd 
import numpy as np 
import yfinance as yf
import plotly.express as px 
from sklearn.pipeline import Pipeline 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler 
from pypfopt.efficient_frontier import EfficientFrontier

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


def multiple_clusterings(n_repeat, data, model, model_name):
    
    '''
      ## Type of data ##
    
      n_repeat : integer --> number of time we apply the clustering method
      model : sklearn model we use --> e.g. GaussianMixture()
      data : pd.DataFrame --> data we want to fit to the model
      model_name : string --> sklearn model name, we use it to create the pipleine

    
      -------------
    
      ## Output ##
    
      Y : a pandas DataFrame object of shape (len(data.index), n_repeat)
      C : a pandas DataFrame object of shpe (n_clusters, n_repeat)
    
      -------------
    
      ## Genera idea ##
    
      The idea is here to train the model on the dataset data multiple time (here n_repeat time)
      and create a DataFrame whose columns are the cluster labels of each stock and whose rows are
      the label of a given stock for each clustering method
    '''

    
    
    pipeline = Pipeline([
    ('scaler', StandardScaler()),   # Étape de standardisation
    (model_name, model) 
    ])

    Y = pd.DataFrame(index=data.index)
    

    dict_centroids = {}
    for i in range(n_repeat):
        pipeline.fit(data)
        predicted_labels = pipeline.named_steps[model_name].labels_
        centroids = pipeline.named_steps[model_name].cluster_centers_.tolist()
        
        data_with_clusters = pd.DataFrame(predicted_labels, index=data.index)
        
        y_i = "Clustering n°%i" % (i+1)
        Y[y_i] = data_with_clusters
        dict_centroids[y_i] = centroids
    C = pd.DataFrame(dict_centroids, index = ["Cluster %i" % (i+1) for i in range(5)])
    
    
    

    return Y, C


def cluster_composition(multiple_clustering):

    n_clustering = len(multiple_clustering.transpose())  ## minus 1 because we don't want to take into account the 
                                                            ## first column that corresponds to the tickers name

    names = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'] ## MODIFIER, C'EST MOCHE

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

def cluster_weights(cluster, centroid, data):
    
    '''
    ----------------------------------------------------------------------
    GENERAL IDEA : Compute the distance from the centre of the cluster 
                    to each stcoks, the disatnce is the eucledian distance 
                    and the weights are the inverse of the distances 
    
    ----------------------------------------------------------------------
    Input : cluster, centroide and the Data
    cluster : list of list and each list is a stocks in the cluster
    centroid : a list wich represent the center of the given cluster 
    ----------------------------------------------------------------------
    output : 

    DataFrame of the weights shape (1, n_stocks_in_cluster) 
    
    '''
    weights = []
    for stock in cluster :
        distance = np.linalg.norm(np.array(centroid)- np.array(data.loc[stock])) # euclidean distance between the center and the stock 
        weight = 1/distance
        weights.append(weight)
              
    weights_matrix = pd.DataFrame(np.array(weights)/sum(weights)).transpose() # we standardize  the weights 

    return weights_matrix



def gaussian_weights(cluster, centroid, data):

    '''
    ----------------------------------------------------------------------
    GENERAL IDEA : Compute the distance from the centre of the cluster 
                    to each stcoks, the disatnce is the eucledian distance 
                    and the weights are this time the gaussian weights. 
                    The exponential allows weights to be lowered more 
                    rapidly as distance increases. 
                    The gaussian weights formula take a standard deviation 
                    as an argument that we chose by trying some values. 
    
    ----------------------------------------------------------------------
    Input : cluster, centroide and the Data
    cluster : list of list and each list is a stocks in the cluster
    centroid : a list wich represent the center of the given cluster 
    ----------------------------------------------------------------------
    output : 

    DataFrame of the weights shape (1, n_stocks_in_cluster) 
    
    '''
    



    
    scaler = StandardScaler()

    normalized_data = scaler.fit_transform(data.loc[cluster]) # we scale the data 
    weights = []
    for stock in cluster:  
        d = np.linalg.norm(np.array(centroid)- np.array(data.loc[stock])) # euclidean distance between the center and the stock 
        weight = np.exp(- d**2/2) # Gaussian weights formula with a deviation of 2 
        weights.append(weight)


    return pd.DataFrame(np.array(weights)/sum(weights)).transpose() #  we standardize  the weights 


def clustering_return(clustering_composition, clustering_composition_centroid, return_data):
    '''
    ----------------------------------------------------------------
    GENERAL IDEA : each cluster is seen as a new asset and the goal 
                   of this routine is to compute the return of this 
                   asset (cluster) given its compositions and weights 
                   put on the different sub-assets that compose this 
                   cluster
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    PARAMS : 
    
    - cluster : pandas dataframe composed of tickers (strings) corresponding to the stocks 
                that compose this cluster 
                [shape : (1, n_stocks_in_cluster)]
                
    
    - weights : pandas dataframe composed of floats that correspond to the weights of each 
                tickers in the cluster
                !! Note that the i-th component of the weights list 
                corresponds to the weight of the i-th ticker in the 
                list cluster !! 
                [shape : (1, n_stocks_in_cluster)]

                
    - return_data : pandas dataframe containing the return of the 
                    stocks 
                    [shape : (n_stocks_in_cluster, n_days_observed)]
    ----------------------------------------------------------------
    '''

    ## We first get back the number of clusters and the 
    ## number of time we repeated the clustering
    n_cluster =  len(clustering_composition.index)


    ## We iterate on the number of clusterings and the 
    ## number of clusters

    for i in range(n_cluster):

        ## We consider the cluster and the centroid
        ## which corresponds to it
        cluster = clustering_composition.iloc[i]

        centroid = clustering_composition_centroid.iloc[i]

        ## Notice that we can also consider gaussian weights
        weights_L2 = cluster_weights(list(cluster), list(centroid), return_data)
                
        ## We use the tickers to get back the returns corresponding to 
        ## the stocks in the cluster 
        cluster_data = return_data.loc[cluster]

        ## We now want to multiply each columns of cluster_data 
        ## by its corresponding weights, here are the steps

        # 1 - Convert DataFrames to NumPy arrays for efficient computation
        array_cluster_data = cluster_data.to_numpy()
        array_weights_gaussian = weights_L2.to_numpy()

        # Transpose B to make it a shape (238, 1)
        array_weights_gaussian = array_weights_gaussian.T


        # Perform the Euclidean scalar product for each column in A and B
        result = np.sum(array_cluster_data * array_weights_gaussian, axis=0)

        
        result_df = pd.DataFrame(result, columns=[clustering_composition.index[i]])

        return result_df

    
    

def cluster_portfolio_return(cluster_composition, weights_matrix, return_data):
    '''
    ----------------------------------------------------------------
    GENERAL IDEA : compute the return of a portfolio composed of 
                   assets (which are clusters) by using the 
                   cluster_return function.
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    PARAMS : 

    - cluster_composition : pandas dataframe. Corresponds to ONE ROW
                            of the dataframe returned by the 
                            cluster_composition function
                            [shape : (n_clusters, 1)]
                
    
    - weights_matrix : pandas dataframe, the weights are distributed in 
                       the same order as the stock_symbols list
                       [shape (1, n_stocks_in_cluster)]

                
    - return_data : pandas dataframe containing the return of the 
                    stocks 
                    [shape (n_stocks_in_cluster, n_days_observed)]
    ----------------------------------------------------------------
    '''
    
    n_clusters = cluster_composition.shape 

    stock_symbols = list(return_data.index)
    
    micro_portfolio_return = pd.DataFrame(index=cluster_composition.index, columns=return_data.columns).transpose()
    
    for i in range(n_clusters):
        cluster = cluster_composition[cluster_composition.index[i]] ## get all the tickers in one cluster
        
        coordonnee_tickers = [stock_symbols.index(element) for element in cluster]

        weight_cluster = pd.DataFrame(weights_matrix[coordonnee_tickers])

        micro_portfolio_return[cluster_composition.index[i]] = cluster_return(cluster, weight_cluster, return_data).transpose()
        
    return micro_portfolio_return.transpose()


def markowitz(expected_returns, cov_matrix):
    """
    Function to obtain the optimized portfolio based on the Sharpe ratio.

    Parameters:
    - expected_returns : Expected returns for each asset.
    - cov_matrix : Covariance matrix of asset returns.

    Returns:
    - clean_weights (dict) : Optimized weights for each asset.
    """

    # Optimize for the maximum Sharpe ratio
    ef = EfficientFrontier(expected_returns, cov_matrix)
    ef.max_sharpe()
    clean_weights = ef.clean_weights()

    return clean_weights

def portfolio_pnl_sharpe(clusters_returns, weights, risk_free_rate=0.03):
    """
    Computes the PnL and Sharpe ratio for a given portfolio composition.

    Parameters:
    - clusters_returns : DataFrame of asset returns where each column represents a cluster and each row a time period.
    - weights (dict): Dictionary of cluster weights (obtained with markowitz). Key is cluster name, value is the weight.
    - risk_free_rate (float): Annualized risk-free rate. Default is 0.03 (3%).

    Returns:
    - pnl (pd.Series): Cumulative PnL of the portfolio.
    - sharpe_ratio (float): Sharpe ratio of the portfolio.
    """
    
    # Calculate the daily portfolio return
    portfolio_returns = clusters_returns.dot(pd.Series(weights))

    # Calculate cumulative PnL
    pnl = (portfolio_returns + 1).cumprod()

    # Calculate Sharpe Ratio
    expected_portfolio_return = portfolio_returns.mean() * 252 # Annualize daily mean return
    portfolio_std_dev = portfolio_returns.std() * np.sqrt(252)  # Annualize daily standard deviation
    sharpe_ratio = (expected_portfolio_return - risk_free_rate) / portfolio_std_dev

    return pnl, sharpe_ratio

        
































