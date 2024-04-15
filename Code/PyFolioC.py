import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import warnings
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------

try:

    from pypfopt.efficient_frontier import EfficientFrontier

except ImportError:

    print("PyPortfolioOpt package not found. Installing...")

    try:

        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPortfolioOpt"])
        from pypfopt.efficient_frontier import EfficientFrontier
        
    except Exception as e:
        print(f"Error installing PyPortfolioOpt package: {e}")
        sys.exit(1)

# ----------------------------------------------------------------

try:

    from signet.cluster import Cluster

except ImportError:

    print("Signet package not found. Installing...")

    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/alan-turing-institute/SigNet.git"])
        from signet.cluster import Cluster

    except Exception as e:
        print(f"Error installing Signet package: {e}")
        sys.exit(1)

# ----------------------------------------------------------------


class PyFolio:



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

    - cov_method : String corresponding to the method we use in for the covariance estimation/construction.

    =================================================================================================================================
    #################################################################################################################################
    =================================================================================================================================
    '''

    def __init__(self, historical_data, lookback_window, evaluation_window, number_of_clusters, sigma, eta, beta, EWA_cov = False, short_selling=False, cov_method='SPONGE', markowitz_type='min_variance'):
        self.historical_data = historical_data
        self.lookback_window = lookback_window
        self.evaluation_window = evaluation_window
        self.number_of_clusters = number_of_clusters
        self.cov_method = cov_method ## 'SPONGE', 'signed_laplacian', 'SPONGE_sym', 'Kmeans'
        self.sigma = sigma
        self.eta = eta
        self.beta = beta
        self.markowitz_type = markowitz_type
        self.EWA_cov = EWA_cov

        self.short_selling = short_selling
        self.correlation_matrix = self.corr_matrix()
        self.cluster_composition = self.cluster_composition_and_centroid()
        self.constituent_weights_res = self.constituent_weights()
        self.cluster_returns = self.cluster_return(lookback_window)

        self.markowitz_weights_res = self.markowitz_weights()
        self.final_weights = self.final_W()

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
        
        ## On respecte le format imposé par signet. Pour cela il faut changer le type des matrices A_pos et A_neg, qui ne peuvent pas rester des dataframes 

        A_pos, A_neg = self.correlation_matrix.applymap(lambda x: x if x >= 0 else 0), self.correlation_matrix.applymap(lambda x: abs(x) if x < 0 else 0)

        data = (sparse.csc_matrix(A_pos.values), sparse.csc_matrix(A_neg.values))

        cluster = Cluster(data)

        ## On applique la méthode SPONGE : clusters the graph using the Signed Positive Over Negative Generalised Eigenproblem (SPONGE) clustering.

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
        
        ## On respecte le format imposé par signet. Pour cela il faut changer le type des matrices A_pos et A_neg, qui ne peuvent pas rester des dataframes 

        A_pos, A_neg = self.correlation_matrix.applymap(lambda x: x if x >= 0 else 0), self.correlation_matrix.applymap(lambda x: abs(x) if x < 0 else 0)

        A_pos_sparse = sparse.csc_matrix(A_pos.values)
        A_neg_sparse = sparse.csc_matrix(A_neg.values)

        data = (A_pos_sparse, A_neg_sparse)

        cluster = Cluster(data)

        ## On applique la méthode SPONGE : clusters the graph using the Signed Positive Over Negative Generalised Eigenproblem (SPONGE) clustering.

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
        
        ## On respecte le format imposé par signet. Pour cela il faut changer le type des matrices A_pos et A_neg, qui ne peuvent pas rester des dataframes 

        A_pos, A_neg = self.correlation_matrix.applymap(lambda x: x if x >= 0 else 0), self.correlation_matrix.applymap(lambda x: abs(x) if x < 0 else 0)

        A_pos_sparse = sparse.csc_matrix(A_pos.values)
        A_neg_sparse = sparse.csc_matrix(A_neg.values)

        data = (A_pos_sparse, A_neg_sparse)

        cluster = Cluster(data)

        ## On applique la méthode SPONGE : clusters the graph using the Signed Positive Over Negative Generalised Eigenproblem (SPONGE) clustering.

        return cluster.SPONGE_sym(self.number_of_clusters)


    def apply_kmeans(self): 

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
        
        data = self.correlation_matrix

        kmeans = KMeans(n_clusters=self.number_of_clusters)
        kmeans.fit(data)
        
        labels = kmeans.labels_

        ## On applique la méthode K-Means.

        return labels
    
    '''
    ###################################################### ATTRIBUTES CONSTRUCTION ######################################################


    In the following section, we provide the code of the routines that are used to construct the main attributes of a portfolio. In the next 
    section, we combine all these routines into a single one named .training_phase()

    '''

    def corr_matrix(self):


        '''
        ----------------------------------------------------------------
        GENERAL IDEA : compute the correlation matrix of different stock 
                    returns  over a given lookback_window
        ----------------------------------------------------------------

        ----------------------------------------------------------------
        PARAMS : 
        
        - lookback_window : list of length 2, [start, end] corresponding 
                            to the range of the lookback_window

        - df_cleaned : pandas dataframe containing the returns of the stocks

        ----------------------------------------------------------------
        '''
    

        correlation_matrix = self.historical_data.iloc[self.lookback_window[0]:self.lookback_window[1], :].corr(method='pearson') ## MODIFIÉ

        correlation_matrix = correlation_matrix.fillna(0) ## in case there are NaN values, we replace them with 0 

        return correlation_matrix


    ## we compute the return_centroid of each cluster to attribute intra-cluster weights according to the distance between stocks within the cluster and this 
    ## centroid

    def cluster_composition_and_centroid(self):

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

        ## STEP 1: run the SPONGE clustering algorithm with a number of clusters fixed to be number_of_clusters and using 
        ##         the correlation matrix correlation_matrix ==> we store the results in result

        ### 1 + pd.DataFrame(...) because we want the number of clusters to range between 1 un number_of_clusters
        if self.cov_method == 'SPONGE':
            result = 1 + pd.DataFrame(index=list(self.correlation_matrix.columns), columns=['Cluster label'], data=self.apply_SPONGE())

        if self.cov_method == 'signed_laplacian':
            result = 1 + pd.DataFrame(index=list(self.correlation_matrix.columns), columns=['Cluster label'], data=self.apply_signed_laplacian())

        if self.cov_method == 'SPONGE_sym':
            result = 1 + pd.DataFrame(index=list(self.correlation_matrix.columns), columns=['Cluster label'], data=self.apply_SPONGE_sym())

        if self.cov_method == 'Kmeans':
            result = 1 + pd.DataFrame(index=list(self.correlation_matrix.columns), columns=['Cluster label'], data=self.apply_kmeans())


        ## STEP 2: compute the composition of each cluster (in terms of stocks)

        cluster_composition = {}

        for i in range(1, self.number_of_clusters + 1):
            if i in result['Cluster label'].values:
                tickers = list(result[result['Cluster label'] == i].index)

                return_centroid = np.zeros(self.lookback_window[1] - self.lookback_window[0])

                for elem in tickers:
                    return_centroid = return_centroid + self.historical_data.loc[:, elem][self.lookback_window[0]:self.lookback_window[1]].values

                centroid = return_centroid / len(tickers)

                cluster_composition[f'cluster {i}'] = {'tickers': tickers, 'centroid': centroid}

        return cluster_composition



    def constituent_weights(self): ## sigma corresponds to some dispersion cofficient
        
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
        OUTPUT : modifies in-place the numpy ndarray returned by the 
                cluster_composition_and_centroid function
        ----------------------------------------------------------------
        '''

        constituent_weights = {}

        for cluster in self.cluster_composition.keys():
            weights = []
            total_cluster_weight = 0

            for elem in self.cluster_composition[cluster]['tickers']:

                elem_returns = self.historical_data.loc[:, elem][self.lookback_window[0]:self.lookback_window[1]].values

                ## we compute the distance of the stock to the centroid of the cluster
                distance_to_centroid = np.linalg.norm(self.cluster_composition[cluster]['centroid'] - elem_returns)**2 

                ## we compute the norm exp(-|x|^2/2*sigma^2)
                total_cluster_weight += np.exp(-distance_to_centroid / (2 * (self.sigma**2)))

                weights.append(np.exp(-distance_to_centroid / (2 * (self.sigma**2))))

            normalized_weights = [w / total_cluster_weight for w in weights]
            constituent_weights[cluster] = dict(zip(self.cluster_composition[cluster]['tickers'], normalized_weights))

        return constituent_weights



    def cluster_return(self, lookback_window):

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

        cluster_returns = pd.DataFrame(index = self.historical_data.index[lookback_window[0]:lookback_window[1]], columns= [f'cluster {i}' for i in range(1, len(self.constituent_weights_res) + 1)], data = np.zeros((len(self.historical_data.index[lookback_window[0]:lookback_window[1]]), len(self.constituent_weights_res))))

        for cluster in self.constituent_weights_res.keys():

            for ticker, weight in self.constituent_weights_res[cluster].items(): 

                ## we transpose df_cleaned to have columns for each ticker
                cluster_returns[cluster] = cluster_returns[cluster] + self.historical_data[ticker][lookback_window[0]:lookback_window[1]]*weight

        return cluster_returns



    """def noised_array(self):

        '''
        ----------------------------------------------------------------
        GENERAL IDEA : given an array y and a target correlation eta, 
                    compute the array with the noise  
        ----------------------------------------------------------------

        ----------------------------------------------------------------
        PARAMS : 

        - y : numpy ndarray that we want to perturb

        - eta : target correlation that we want to create between y and 
                its perturbated version

        ----------------------------------------------------------------

        ----------------------------------------------------------------
        OUTPUT : noised version of y that satisfies the targeted level 
                of correlation
        ----------------------------------------------------------------
        '''
        
        # We compute with a small noise 
        epsilon_std_dev = 0.001

        # Calculer la corrélation initiale

        correlation = 1

        if self.cov_method == 'forecast':
            y = self.historical_data.iloc[self.lookback_window[1]: self.lookback_window[1]+self.evaluation_window,:].mean()

        else:
            y = self.cluster_return(lookback_window=[self.lookback_window[1], self.lookback_window[1]+self.evaluation_window]).mean()

        x = y.copy()

        # Boucle pour ajuster l'écart-type du bruit jusqu'à ce que la corrélation atteigne eta

        while correlation > self.eta:

            # Generate a vector of Gaussian noise
            noise = np.random.normal(0, epsilon_std_dev, len(y))

            x = noise + y

            correlation = x.corr(y.squeeze())

            # Adjust the standard deviation of the noise
            epsilon_std_dev += 0.0005

        return x"""

    def noised_array(self):

        if self.eta==0: ## si eta = 0, expected_return = moyenne des returns sur la période d'entrainement

                return(self.cluster_return(self.lookback_window).mean()) ## si eta = 0, on choisit l'approche naïve qui consiste à ne pas 
                                                                         ## introduire du tout d'alpha et de prédiction
            
        else:
            # Extraction des rendements des actifs sur la période d'évaluation

            asset_returns = self.cluster_return(lookback_window=[self.lookback_window[1], self.lookback_window[1]+self.evaluation_window])

            if self.eta==1:

                return(self.cluster_return(lookback_window=[self.lookback_window[1], self.lookback_window[1]+self.evaluation_window]).mean())
            
            else:
                # Calcul des moyennes et des écarts-types des rendements pour chaque actif
                asset_means = asset_returns.mean()
                asset_std_devs = asset_returns.std()

                # Initialisation du DataFrame pour stocker les rendements bruités
                noised_returns = asset_means.copy()

                # Itération sur chaque colonne (actif) pour ajouter du bruit
                for asset in asset_means.index:
                    # Calcul de l'écart-type du bruit pour cet actif
                    noise_std_dev = np.sqrt(asset_std_devs[asset]*2 / self.eta - asset_std_devs[asset]*2)

                    # Génération du bruit
                    noise = np.random.normal(0, noise_std_dev)

                    # Ajout du bruit aux rendements de l'actif
                    noised_returns[asset] = asset_means[asset] + noise

                return noised_returns
    


    def markowitz_weights(self):

        '''
        ----------------------------------------------------------------
        GENERAL IDEA : compute the markowitz weights of each cluster in 
                    the synthetic portfolio using the pypfopt package
        ----------------------------------------------------------------

        ----------------------------------------------------------------
        PARAMS : 

        - cluster_return : numpy array as returned by the 
                        cluster_return function 

        - df_cleaned : pandas dataframe containing the returns of the 
                    stocks

        - constituent_weights : numpy array as returned by the 
                                constituent_weights function 

        - lookback_window : list of length 2, [start, end] corresponding 
                            to the range of the lookback_window

        - evaluation_window : integer, corresponding to the number of 
                            days that we look bakc at to make our 
                            prevision

        - eta : target correlation that we want to create between y and 
                its perturbated version
        ----------------------------------------------------------------

        ----------------------------------------------------------------
        OUTPUT : returns the markowitz weights of each cluster
        ----------------------------------------------------------------
        '''

        if self.EWA_cov:

            X = self.cluster_returns.transpose()
            _, n_days = X.shape
            cov = ((1 - self.beta)/(1 - self.beta ** n_days)) * sum((self.beta**(n_days - t)*np.outer(X.iloc[:, t - 1].values, X.iloc[:, t - 1].values)) for t in range(1, n_days + 1)).transpose()
            cov = pd.DataFrame(index=self.cluster_returns.columns, columns=self.cluster_returns.columns, data=cov)

        else:  
            cov = self.cluster_returns.cov()

        cov = cov.fillna(0.)

        ## on construit le vecteur d'expected return du cluster (252 jours de trading par an, on passe de rendements journaliers à rendements annualisés)
                
        expected_returns = self.noised_array()

        if self.short_selling: ## if we allow short-selling, then weights are not constrained to take nonnegative values, 
                               ## hence the (-1, 1) bounds
        
            ef = EfficientFrontier(expected_returns=expected_returns, cov_matrix=cov, weight_bounds=(-1, 1)) 
        
        else: 

            ef = EfficientFrontier(expected_returns=expected_returns, cov_matrix=cov, weight_bounds=(0, 1))

        if self.markowitz_type == 'min_variance':

            ef.min_volatility()

        elif self.markowitz_type == 'max_sharpe':
            ef.max_sharpe(risk_free_rate=0)
        
        elif self.markowitz_type == 'expected_returns':
            ef.efficient_return(target_return=expected_returns)

        markowitz_weights = ef.clean_weights()

        return markowitz_weights


    def final_W(self):

        '''
        ----------------------------------------------------------------
        GENERAL IDEA : compute the final weights of each individual stock
                    in the overal portfolio using both the constituent 
                    and the markowitz weights
        ----------------------------------------------------------------

        ----------------------------------------------------------------
        PARAMS : 

        - markowitz_weights : numpy array as returned by the 
                            markowitz_weights function 

        - constituent_weights : integer, corresponding to the number of lookback 
                                days (in terms of historcal returns)
        ----------------------------------------------------------------

        ----------------------------------------------------------------
        OUTPUT : returns the final weights of each asset, i.e. the 
                overall portfolio weights
        ----------------------------------------------------------------
        '''

        ### On cherche désormais à calculer le poids de chaque actif dans le portefeuille total

        W = {}

        for cluster in self.constituent_weights_res.keys(): ## we range across all clusters

            for tickers, weight in self.constituent_weights_res[cluster].items(): ## we range across all tickers in each cluster

                W[tickers] = weight*self.markowitz_weights_res[cluster]

        W = pd.DataFrame(list(W.items()), columns=['ticker', 'weights'])
    
        W.set_index('ticker', inplace=True)

        return W
    
 
class PyFolioC(PyFolio):

    def __init__(self, number_of_repetitions, historical_data, lookback_window, evaluation_window, number_of_clusters, sigma, eta, beta, EWA_cov = False, short_selling=False, cov_method='SPONGE', transaction_cost_rate=0.0001):
        
        super().__init__(historical_data, 
                         lookback_window, 
                         evaluation_window, 
                         number_of_clusters, 
                         sigma, 
                         eta, 
                         beta, 
                         EWA_cov,
                         short_selling, 
                         cov_method)
        
        self.number_of_repetitions = number_of_repetitions
        self.transaction_cost_rate = transaction_cost_rate
        self.previous_weights = None 
        self.consolidated_weight = self.consolidated_W()
        self.portfolio_return = self.portfolio_returns()

    def consolidated_W(self):

        '''
        ----------------------------------------------------------------
        GENERAL IDEA : consolidate the numpy array of weights by 
                    repeating the training and portfolio construction
                    phase a certain number of times 
                    (number_of_repetitions).
        ----------------------------------------------------------------

        ----------------------------------------------------------------
        PARAMS : 

        - number_of_repetitions : number of time we repeat the training
                                phase and the consequent averaging 
                                method

        - lookback_window : list of length 2, [start, end] corresponding 
                            to the range of the lookback_window

        - df_cleaned : cleaned pandas dataframe containing the returns 
                    of the stocks

        - number_of_clusters : integer, corresponding to the number of 
                            clusters

        - sigma : float, corresponding to the dispersion in the intra-
                cluster weights

        - df : pandas dataframe containing the raw data

        ----------------------------------------------------------------

        ----------------------------------------------------------------
        OUTPUT : numpy ndarray containing the returns of the overall weights of each cluster
        ----------------------------------------------------------------
        '''

        # Initialize an empty DataFrame to store the results
        consolidated_W = pd.DataFrame()

        # Run the training function n times and concatenate the results
        for _ in range(self.number_of_repetitions):

            # Assuming training() returns a DataFrame with 'weights' as the column name
            portfolio = PyFolio(historical_data=self.historical_data, lookback_window=self.lookback_window, evaluation_window=self.evaluation_window, number_of_clusters=self.number_of_clusters, sigma=self.sigma, eta=self.eta, beta=self.beta, EWA_cov=self.EWA_cov, short_selling=self.short_selling, cov_method=self.cov_method)

            weights_df = portfolio.final_weights

            # Concatenate the results into columns
            consolidated_W = pd.concat([consolidated_W, weights_df], axis=1)

        # Calculate the average along axis 1
        average_weights = consolidated_W.mean(axis=1)

        # Create a DataFrame with the average weights
        consolidated_W = pd.DataFrame({'weight': average_weights})

        consolidated_W = consolidated_W.transpose()

        return consolidated_W


    def portfolio_returns(self):

        '''
        ----------------------------------------------------------------
        GENERAL IDEA : given the overall weights of each asset in the 
                    portfolio, compute the portfolio return over an 
                    evaluation window that does not overlap with the 
                    lookback_window. 
        ----------------------------------------------------------------

        ----------------------------------------------------------------
        PARAMS : 

        - evaluation_window : integer, corresponding to the number of 
                            future days (in terms of historcal returns) 
                            on which we evaluate the portfolio

        - lookback_window : list of length 2, [start, end] corresponding 
                            to the range of the lookback_window

        - df_cleaned : cleaned pandas dataframe containing the returns 
                    of the stocks

        - consolidated_W : numpy ndarray, containing the final weights 
                        of each asset, i.e. the overall portfolio 
                        weights

        - df : pandas dataframe containing the raw data
        ----------------------------------------------------------------

        ----------------------------------------------------------------
        OUTPUT : returns the portfolio return of each cluster in a 
                pandas dataframe
        ----------------------------------------------------------------
        '''

        portfolio_returns = pd.DataFrame(index=self.historical_data.iloc[self.lookback_window[1]:self.lookback_window[1]+self.evaluation_window, :].index, columns=['return'], data=np.zeros(len(self.historical_data.iloc[self.lookback_window[1]:self.lookback_window[1]+self.evaluation_window, :].index)))

        for ticker in self.consolidated_weight.columns: 

        ##  each time we add :            the present value of the return + the weighted "contribution" of the stock 'ticker' times is weight in the portfolio
            portfolio_returns['return'] = portfolio_returns['return'] + self.historical_data[ticker][self.lookback_window[1]:self.lookback_window[1]+self.evaluation_window]*self.consolidated_weight[ticker]['weight']

        return portfolio_returns
    

    def sliding_window(self, number_of_window, include_transaction_costs=True):
    
        PnL = []
        daily_PnL = []
        overall_return = pd.DataFrame()
        portfolio_value=[1] #we start with a value of 1, the list contain : the porfolio value at the start of each evaluation period
        lookback_window_0 = self.lookback_window
        for i in range(1, number_of_window + 1):

            consolidated_portfolio = PyFolioC(number_of_repetitions=self.number_of_repetitions, historical_data=self.historical_data, lookback_window=lookback_window_0, evaluation_window=self.evaluation_window, number_of_clusters=self.number_of_clusters, sigma=self.sigma, eta=self.eta, beta=self.beta, EWA_cov=self.EWA_cov, short_selling=self.short_selling, cov_method=self.cov_method)
            current_weights= self.consolidated_weight
            if self.previous_weights is None:
                Turnover = 1.0
            else:
                Turnover = np.sum(np.abs(current_weights.squeeze() - self.previous_weights.squeeze()))
            transaction_costs = Turnover*self.transaction_cost_rate if include_transaction_costs else 0
            overall_return = pd.concat([overall_return, consolidated_portfolio.portfolio_return -transaction_costs / self.evaluation_window])

            adjusted_returns = consolidated_portfolio.portfolio_return['return'] - (transaction_costs / self.evaluation_window)

            # Now calculate the cumulative product of the adjusted returns
            cumulative_returns = np.cumprod(1 + adjusted_returns) * portfolio_value[-1] - portfolio_value[-1]

            # Reshape the cumulative returns to match the expected evaluation window size and concatenate to the PnL array
            PnL = np.concatenate((PnL, np.reshape(cumulative_returns, (self.evaluation_window,))))
            daily_PnL = np.concatenate((daily_PnL, np.reshape(np.cumprod(1 + consolidated_portfolio.portfolio_return)*portfolio_value[-1] - portfolio_value[-1], (self.evaluation_window,))))
            portfolio_value.append(portfolio_value[-1]+PnL[-1])

            print(f'step {i}/{number_of_window}, portfolio value: {portfolio_value[-1]:.4f}')
            self.previous_weights = current_weights
            lookback_window_0 = [self.lookback_window[0] + self.evaluation_window*i, self.lookback_window[1] + self.evaluation_window*i]
        n = len(PnL)//self.evaluation_window

        for j in range(1, n):

            for i in range(1, self.evaluation_window+1):
                
                PnL[j*self.evaluation_window + i - 1] = PnL[j*self.evaluation_window + i - 1] + PnL[j*self.evaluation_window - 1]
        
        return overall_return, PnL, portfolio_value, daily_PnL