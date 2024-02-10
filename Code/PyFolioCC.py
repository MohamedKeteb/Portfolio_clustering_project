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



    def __init__(self, historical_data, lookback_window, evaluation_window, number_of_clusters, clustering_method='SPONGE'):
        self.historical_data = historical_data
        self.lookback_window = lookback_window
        self.evaluation_window = evaluation_window
        self.number_of_clusters = number_of_clusters
        self.clustering_method = clustering_method
        self.correlation_matrix = self.corr_matrix()


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
        A_pos, A_neg = self.correlation_matrix.applymap(lambda x: x if x >= 0 else 0), self.corlation_matrix.applymap(lambda x: abs(x) if x < 0 else 0)

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
        A_pos, A_neg = self.correlation_matrix.applymap(lambda x: x if x >= 0 else 0), self.corlation_matrix.applymap(lambda x: abs(x) if x < 0 else 0)

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
        A_pos, A_neg = self.correlation_matrix.applymap(lambda x: x if x >= 0 else 0), self.corlation_matrix.applymap(lambda x: abs(x) if x < 0 else 0)

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
    
    