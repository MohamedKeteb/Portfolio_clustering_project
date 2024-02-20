import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

## path Nail : '/Users/khelifanail/Documents/GitHub/Portfolio_clustering_project'
## path Jerome : 'C:/Users/33640/OneDrive/Documents/GitHub/Portfolio_clustering_project'
sys.path.append(r'/Users/khelifanail/Documents/GitHub/Portfolio_clustering_project')  # Ajoute le chemin parent

from signet.cluster import Cluster 
from scipy import sparse
from pypfopt.efficient_frontier import EfficientFrontier




class PyFolioCC:



    '''
    ================================================================================================================================
    ######################################################## DOCUMENTATION #########################################################
    ================================================================================================================================

    --------------------------------------------------------- INTRODUCTION ---------------------------------------------------------
    
    The PyFolioCC class is designed to build an optimal portfolio in the sense of Markowitz using general graph clustering 
    techniques. The idea is to provide a historical return database of an asset universe (historical_data), a lookback window 
    (lookback_window) for portfolio construction, a number of clusters (number_clusters), a clustering method (clustering_method), 
    and an evaluation window (evaluation_window). From there, the objective is to construct a portfolio based on historical return 
    data over the period corresponding to lookback_window by creating a sub-portfolio composed of a specified number of synthetic 
    assets (ETFs) using the clustering method specified in clustering_method. The performance (Sharpe ratio and cumulative PnL) of
    the constructed portfolio is then evaluated over the evaluation_window.

    ---------------------------------------------------------- PARAMETERS ----------------------------------------------------------
    
    - historical_data : Pandas DatFrame of shape (n_assets, n_days). The indices must be asset tickers ('AAPL' for Apple, 'MSFT' 
                        for Microsoft...).

    - lookback_window : List of length 2 [starting_day, final_day]. For instance, if the lookback_window is [0, 252] this means that
                        we construct the portfolio on the first trading year of historical return 
                        (i.e. on historical_data.iloc[:, lookback_window[0]:lookback_window[1]]).

    - evaluation_window : Integer corresponding to the number of days on which to evaluate the performance of the portfolio. 

    - number_of_clusters : Integer corresponding to the number of clusters in which we split the portfolio. 

    - clustering_method : String corresponding to the clustering method that we use in the portfolio construction phase.

    =================================================================================================================================
    #################################################################################################################################
    =================================================================================================================================
    '''



    def __init__(self, historical_data, lookback_window, evaluation_window, number_of_clusters, sigma, clustering_method='SPONGE'):
        self.historical_data = historical_data
        self.lookback_window = lookback_window
        self.evaluation_window = evaluation_window
        self.number_of_clusters = number_of_clusters
        self.clustering_method = clustering_method
        self.correlation_matrix = self.corr_matrix()
        self.sigma = sigma
        self.cluster_returns = self.cluster_return()


    '''
    ###################################################### CLUSTERING METHODS ######################################################


    In the following section, we provide the code to apply three clustering methods (SPONGE, Symmetric SPONGE, Signed_Laplacian). 
    These routines are fundamental as they allow obtaining clustering from the correlation matrix of assets in our portfolio.

    '''

    def apply_SPONGE(self): 
        '''
        ----------------------------------------------------------------
        IDEA: Given a correlation matrix obtained from a database and 
              Pearson similarity, return a vector associating each asset 
              with the cluster number it belongs to after applying SPONGE 
              (using the signet package).
        ----------------------------------------------------------------

        ----------------------------------------------------------------
        PARAMS: 

        - self.correlation_matrix: a square dataframe of size 
                                   (number_of_stocks, number_of_stocks)

        - self.number_of_clusters : the number of clusters to identify. 
                                    If a list is given, the output is a 
                                    corresponding list

        ----------------------------------------------------------------

        ----------------------------------------------------------------
        OUTPUT: array of int, or list of array of int: Output assignment 
                to clusters.
        ----------------------------------------------------------------
        '''

        ## We respect the format imposed by signet. To do this, we need to change the type of the A_pos and A_neg matrices, which cannot remain dataframes
        A_pos, A_neg = self.correlation_matrix.applymap(lambda x: x if x >= 0 else 0), self.correlation_matrix.applymap(lambda x: abs(x) if x < 0 else 0)

        data = (sparse.csc_matrix(A_pos.values), sparse.csc_matrix(A_neg.values))

        cluster = Cluster(data)

        ## We apply the SPONGE method: clusters the graph using the Signed Positive Over Negative Generalised Eigenproblem (SPONGE) clustering.
        return cluster.SPONGE(self.number_of_clusters)
    

    def apply_signed_laplacian(self): 
        '''
        ----------------------------------------------------------------
        IDEA: Given a correlation matrix obtained from a database and 
              Pearson similarity, return a vector associating each asset 
              with the cluster number it belongs to after applying the 
              signed Laplacian method (using the signet package).
        ----------------------------------------------------------------

        ----------------------------------------------------------------
        PARAMS: 

        - correlation_matrix: a square dataframe of size 
                              (number_of_stocks, number_of_stocks)

        - self.number_of_clusters: the number of clusters to identify. 
                                   If a list is given, the output is a 
                                   corresponding list

        ----------------------------------------------------------------

        ----------------------------------------------------------------
        OUTPUT: array of int, or list of array of int: Output assignment 
                to clusters.
        ----------------------------------------------------------------
        '''

        ## We respect the format imposed by signet. To do this, we need to change the type of the A_pos and A_neg matrices, which cannot remain dataframes
        A_pos, A_neg = self.correlation_matrix.applymap(lambda x: x if x >= 0 else 0), self.correlation_matrix.applymap(lambda x: abs(x) if x < 0 else 0)

        A_pos_sparse = sparse.csc_matrix(A_pos.values)
        A_neg_sparse = sparse.csc_matrix(A_neg.values)

        data = (A_pos_sparse, A_neg_sparse)

        cluster = Cluster(data)

        ## We apply the signed Laplacian method: clusters the graph using spectral clustering with the signed Laplacian.
        return cluster.spectral_cluster_laplacian(self.number_of_clusters)


    def apply_SPONGE_sym(self): 
        '''
        ----------------------------------------------------------------
        IDEA: Given a correlation matrix obtained from a database and 
              Pearson similarity, return a vector associating each asset 
              with the cluster number it belongs to after applying 
              symmetric SPONGE (using the signet package).
        ----------------------------------------------------------------

        ----------------------------------------------------------------
        PARAMS: 

        - self.correlation_matrix: a square dataframe of size 
                                   (number_of_stocks, number_of_stocks)

        - self.number_of_clusters: the number of clusters to identify. 
                                   If a list is given, the output is a 
                                   corresponding list

        ----------------------------------------------------------------

        ----------------------------------------------------------------
        OUTPUT: array of int, or list of array of int: Output assignment 
                to clusters.
        ----------------------------------------------------------------
        '''

        ## We respect the format imposed by signet. To do this, we need to change the type of the A_pos and A_neg matrices, which cannot remain dataframes
        A_pos, A_neg = self.correlation_matrix.applymap(lambda x: x if x >= 0 else 0), self.correlation_matrix.applymap(lambda x: abs(x) if x < 0 else 0)

        A_pos_sparse = sparse.csc_matrix(A_pos.values)
        A_neg_sparse = sparse.csc_matrix(A_neg.values)

        data = (A_pos_sparse, A_neg_sparse)

        cluster = Cluster(data)

        ## We apply the symmetric SPONGE method: clusters the graph using the Signed Positive Over Negative Generalised Eigenproblem (SPONGE) clustering with symmetry.
        return cluster.SPONGE_sym(self.k)




    def corr_matrix(self):
        '''
        ----------------------------------------------------------------
        GENERAL IDEA: compute the correlation matrix of different stock 
                    returns  over a given lookback_window
        ----------------------------------------------------------------

        ----------------------------------------------------------------
        PARAMS: 
        
        - lookback_window : list of length 2, [start, end] corresponding 
                            to the range of the lookback_window

        - df_cleaned : pandas dataframe containing the returns of the stocks

        ----------------------------------------------------------------

        ----------------------------------------------------------------
        OUTPUT: correlation_matrix of size 
                (number_of_assets, number_of_assets)
        ----------------------------------------------------------------
        '''
    

        correlation_matrix = self.historical_data.iloc[:, self.lookback_window[0]:self.lookback_window[1]].transpose().corr(method='pearson') ## MODIFIÉ

        correlation_matrix = correlation_matrix.fillna(0) ## in case there are NaN values, we replace them with 0 

        return correlation_matrix
    
    
    def cluster_return(self, sigma):

        '''
        ----------------------------------------------------------------
        GENERAL IDEA : 
        1. Get the composition of each cluster (so as to compute the return 
        of each cluster seen as a new asset)
        2. Get the centroid of each cluster (so as to compute intra-cluster
        weights that will be used to compute the overall return of each 
        cluster (with the idea that each stock has a different contribution
        to the overall cluster))
        ----------------------------------------------------------------

        ----------------------------------------------------------------
        PARAMS : 
        
        - df_cleaned : pandas dataframe containing the returns of the 
                    stocks

        - correlation_matrix : pandas dataframe as given by the previous  
                            correlation_matrix function

        - number_of_clusters : integer, corresponding to the number of 
                            clusters

        - lookback_window : list of length 2, [start, end] corresponding 
                            to the range of the lookback_window
        ----------------------------------------------------------------
        '''
        ## cluster composition and centroids

        result = dict(zip(list(self.correlation_matrix.columns), self.apply_SPONGE())) ## composition

        df_cleaned = self.historical_data.copy()

        df_cleaned['Cluster'] = df_cleaned.index.map(result)
        centroid_returns = df_cleaned.groupby('Cluster').mean() ## centroids 

        df_cleaned = df_cleaned.transpose() ## contains the historical returns and a lign that indicates the cluster to which each stock belongs
        centroid_returns = centroid_returns.transpose()

        ## constituent_weights ##

        '''
        ----------------------------------------------------------------
        GENERAL IDEA : compute the constituent weights (i.e.
        the intra-cluster weights of each stock)
        ----------------------------------------------------------------

        ----------------------------------------------------------------
        PARAMS : 
        
        - df_cleaned : pandas dataframe containing the returns of the 
                    stocks

        - cluster_composition : numpy array as returned by the 
                                cluster_composition_and_centroid 
                                function

        - sigma : parameter of dispersion

        - lookback_window : list of length 2, [start, end] corresponding 
                            to the range of the lookback_window
        ----------------------------------------------------------------

        ----------------------------------------------------------------
        OUTPUT : numpy array containing the weights of each stock 
        ----------------------------------------------------------------
        '''

        constituent_weights = pd.DataFrame(index=['Weight'], columns=df_cleaned.columns)
        total_weight = pd.DataFrame(index=['Total weight'], columns=[i for i in range(self.number_of_clusters)], data=np.zeros((1, self.number_of_clusters)))

        ## we first compute the difference between the cluster centroid return and the cluster ticker return
        for ticker in df_cleaned.columns:
            df_cleaned[ticker][:-1] = df_cleaned[ticker][:-1] - centroid_returns[int(df_cleaned[ticker]['Cluster'])]

        ## we use this difference to compute the distance between each asset and its cluster centroid return 
        for ticker in df_cleaned.columns:
            constituent_weights[ticker] = np.exp(self.sigma*((np.linalg.norm(df_cleaned[ticker][:-1]))**2)/2)
            total_weight[int(df_cleaned[ticker]['Cluster'])]['Total weight'] += np.exp(self.sigma*((np.linalg.norm(df_cleaned[ticker][:-1]))**2)/2)

        ## we normalize the weights
        for ticker in df_cleaned.columns:
            constituent_weights[ticker] = constituent_weights[ticker]['Weight']/total_weight[int(df_cleaned[ticker]['Cluster'])]['Total weight']

        ## check whether the weights equal to 0 within each cluster: 
        # constituent_weights[[ticker for ticker in df_cleaned.columns if df_cleaned[ticker]['Cluster']  == 1.0]].sum(axis=1)
        
        '''
        ----------------------------------------------------------------
        GENERAL IDEA : compute the return of each cluster.
                    The steps are : 
                    1. find the assets composing each cluster
                    2. compute the consituent_weights weighted-average 
                    return of all those stocks, which is by definition 
                    the return of the cluster
                    
        ----------------------------------------------------------------

        ----------------------------------------------------------------
        PARAMS : 
        
        - df_cleaned : pandas dataframe containing the returns of the 
                    stocks

        - constituent_weights : numpy array as returned by the 
                                constituent_weights function 

        - lookback_window : list of length 2, [start, end] corresponding 
                            to the range of the lookback_window
        ----------------------------------------------------------------

        ----------------------------------------------------------------
        OUTPUT : create a single column pandas dataframe containing the 
                return of each cluster over the lookback_window days
        ----------------------------------------------------------------
        '''
        cluster_return = pd.DataFrame(index=df_cleaned.index[:-1], columns=np.arange(self.number_of_clusters), data=np.zeros((df_cleaned.shape[0] - 1, self.number_of_clusters))) ## -1 and [:-1] because we don't want to take into account the last line 
        ## that contains the label of the cluster for each stock

        for ticker in df_cleaned.columns:
            cluster_return[int(df_cleaned[ticker]['Cluster'])] = cluster_return[int(df_cleaned[ticker]['Cluster'])] + constituent_weights[ticker]['Weight'] * df_cleaned[ticker][:-1]

        return cluster_return
            
