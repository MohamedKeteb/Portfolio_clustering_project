import numpy as np
import pandas as pd
import ast
import sys
import plotly.graph_objects as go
import matplotlib.pyplot as plt 


## path Nail : '/Users/khelifanail/Documents/GitHub/Portfolio_clustering_project'
## path Jerome : 'C:/Users/33640/OneDrive/Documents/GitHub/Portfolio_clustering_project'
sys.path.append(r'/Users/khelifanail/Documents/GitHub/Portfolio_clustering_project')  # Ajoute le chemin parent

from signet.cluster import Cluster 
from scipy import sparse
from pypfopt.efficient_frontier import EfficientFrontier



# Function to safely convert a string into a list
def safe_literal_eval(s):
    try:
        # Tries to convert the string into a list
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # If an error occurs, returns a default value, e.g. an empty list
        return []


def check_nan_inf(df):
    # Vérification des valeurs NaN
    if df.isna().any().any():
        print("There are NaN values in the dataframe")
    else:
        print("There are no NaN values in the dataframe")


def remove_rows_with_nan(df):
    return df.dropna()



def signed_adjency(mat):
    '''
    L'idée est ici, à partir d'une matrice de corrélation mat, de renvoyer deux matrices 
    A_positive et A_negative qui correspondraient aux matrices des corrélations positives et négatives 
    associées  
    '''

    A_pos = mat.applymap(lambda x: x if x >= 0 else 0)
    A_neg = mat.applymap(lambda x: abs(x) if x < 0 else 0)
    
    return A_pos, A_neg

def apply_SPONGE(correlation_matrix, k): 

    '''
    IDÉE : étant donné une matrice de correlation obtenue à partir d'une base de donnée et de la similarité de pearson, renvoyer un vecteur associant 
           à chaque actif le numéro du cluster auquel il appartient une fois qu'on lui a appliqué SPONGE (à partir du package signet)

    PARAMS : 

    - correlation_matrix : a square dataframe of size (number_of_stocks, number_of_stocks)
    - k : the number of clusters to identify. If a list is given, the output is a corresponding list

    RETURNS : array of int, or list of array of int: Output assignment to clusters.

    '''
    
    ## On respecte le format imposé par signet. Pour cela il faut changer le type des matrices A_pos et A_neg, qui ne peuvent pas rester des dataframes 

    A_pos, A_neg = signed_adjency(correlation_matrix)

    data = (sparse.csc_matrix(A_pos.values), sparse.csc_matrix(A_neg.values))

    cluster = Cluster(data)

    ## On applique la méthode SPONGE : clusters the graph using the Signed Positive Over Negative Generalised Eigenproblem (SPONGE) clustering.

    return cluster.SPONGE(k)

def apply_signed_laplacian(correlation_matrix, k): 

    '''
    IDÉE : étant donné une matrice de correlation obtenue à partir d'une base de donnée et de la similarité de pearson, renvoyer un vecteur associant 
           à chaque actif le numéro du cluster auquel il appartient une fois qu'on lui a appliqué SPONGE (à partir du package signet)

    PARAMS : 

    - correlation_matrix : a square dataframe of size (number_of_stocks, number_of_stocks)
    - k : the number of clusters to identify. If a list is given, the output is a corresponding list

    RETURNS : array of int, or list of array of int: Output assignment to clusters.

    '''
    
    ## On respecte le format imposé par signet. Pour cela il faut changer le type des matrices A_pos et A_neg, qui ne peuvent pas rester des dataframes 

    A_pos, A_neg = signed_adjency(correlation_matrix)

    A_pos_sparse = sparse.csc_matrix(A_pos.values)
    A_neg_sparse = sparse.csc_matrix(A_neg.values)

    data = (A_pos_sparse, A_neg_sparse)

    cluster = Cluster(data)

    ## On applique la méthode SPONGE : clusters the graph using the Signed Positive Over Negative Generalised Eigenproblem (SPONGE) clustering.

    return cluster.spectral_cluster_laplacian(k)

def apply_SPONGE_sym(correlation_matrix, k): 

    '''
    IDÉE : étant donné une matrice de correlation obtenue à partir d'une base de donnée et de la similarité de pearson, renvoyer un vecteur associant 
           à chaque actif le numéro du cluster auquel il appartient une fois qu'on lui a appliqué SPONGE (à partir du package signet)

    PARAMS : 

    - correlation_matrix : a square dataframe of size (number_of_stocks, number_of_stocks)
    - k : the number of clusters to identify. If a list is given, the output is a corresponding list

    RETURNS : array of int, or list of array of int: Output assignment to clusters.

    '''
    
    ## On respecte le format imposé par signet. Pour cela il faut changer le type des matrices A_pos et A_neg, qui ne peuvent pas rester des dataframes 

    A_pos, A_neg = signed_adjency(correlation_matrix)

    A_pos_sparse = sparse.csc_matrix(A_pos.values)
    A_neg_sparse = sparse.csc_matrix(A_neg.values)

    data = (A_pos_sparse, A_neg_sparse)

    cluster = Cluster(data)

    ## On applique la méthode SPONGE : clusters the graph using the Signed Positive Over Negative Generalised Eigenproblem (SPONGE) clustering.

    return cluster.SPONGE_sym(k)

def correlation_matrix(lookback_window, df_cleaned):


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
 

    correlation_matrix = df_cleaned.iloc[:, lookback_window[0]:lookback_window[1]].transpose().corr(method='pearson') ## MODIFIÉ
    return correlation_matrix


## we compute the return_centroid of each cluster to attribute intra-cluster weights according to the distance between stocks within the cluster and this 
## centroid

def cluster_composition_and_centroid(df_cleaned, correlation_matrix, number_of_clusters, lookback_window, clustering_method='SPONGE'):

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

    if clustering_method == 'SPONGE':
        result = pd.DataFrame(index=list(correlation_matrix.columns), columns=['Cluster label'], data=apply_SPONGE(correlation_matrix, number_of_clusters))

    if clustering_method == 'signed_laplacian':
        result = pd.DataFrame(index=list(correlation_matrix.columns), columns=['Cluster label'], data=apply_signed_laplacian(correlation_matrix, number_of_clusters))

    if clustering_method == 'SPONGE_sym':
        result = pd.DataFrame(index=list(correlation_matrix.columns), columns=['Cluster label'], data=apply_SPONGE_sym(correlation_matrix, number_of_clusters))


    ## STEP 2: compute the composition of each cluster (in terms of stocks)

    cluster_composition = []

    for i in range(1, number_of_clusters):

        if i in result['Cluster label'].values: ## we check that the i-th cluster is not empty

            cluster_composition.append([f'cluster {i}', list(result[result['Cluster label'] == i].index)])

    ## STEP 3: compute the centroid of each cluster 

    for i in range(len(cluster_composition)):

        return_centroid = np.zeros(lookback_window[1]-lookback_window[0]) ## we prepare the return_centroid array to stock the centroid

        for elem in cluster_composition[i][1]:

            return_centroid = return_centroid + df_cleaned.loc[elem, :][lookback_window[0]:lookback_window[1]].values

        cluster_composition[i].append(return_centroid/len(cluster_composition[i][1])) ## the third element contains the centroid of the cluster in question

    return cluster_composition



def constituent_weights(df_cleaned, cluster_composition, sigma, lookback_window): ## sigma corresponds to some dispersion cofficient
    
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

    constituent_weights = []

    for i in range(len(cluster_composition)): ## we range across all clusters 

        weights = []
        total_cluster_weight = 0 ## we store the total weights within a cluster to normalize weights in the end

        for elem in cluster_composition[i][1]:

            elem_returns = df_cleaned.loc[elem, :][lookback_window[0]:lookback_window[1]].values

            distance_to_centroid = np.linalg.norm(cluster_composition[i][2] - elem_returns)**2
            
            total_cluster_weight += np.exp(-distance_to_centroid/(2*(sigma**2)))

            weights.append([elem, np.exp(-distance_to_centroid/(2*(sigma**2)))])

        for j in range(len(weights)):
            weights[j][1] = weights[j][1]/total_cluster_weight
                
        constituent_weights.append([cluster_composition[i][0], weights])

    return constituent_weights


def cluster_return(constituent_weights, df_cleaned, df, lookback_window):

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

    ## we first get the open and close values for each stock 
    open = pd.DataFrame(index = df_cleaned.index, columns=df_cleaned.columns[lookback_window[0]:lookback_window[1]])
    close = pd.DataFrame(index = df_cleaned.index, columns=df_cleaned.columns[lookback_window[0]:lookback_window[1]])

    for stock in open.index:
        open.loc[stock, :] = df.loc[stock, 'open'][lookback_window[0]:lookback_window[1]]
        close.loc[stock, :] = df.loc[stock, 'close'][lookback_window[0]:lookback_window[1]]

    ## using open and close, we compute the returns of each stocks (weighted-average using constituents weights)
    cluster_returns = pd.DataFrame(index = [f'cluster {i+1}' for i in range(len(constituent_weights))], columns = df_cleaned.columns[lookback_window[0]:lookback_window[1]])

    for returns in cluster_returns.columns:

        for elem in constituent_weights:
            op, cl = 0, 0
            for stocks in elem[1]:
                op += open.loc[stocks[0], returns]*stocks[1]
                cl += close.loc[stocks[0], returns]*stocks[1]

            cluster_returns.loc[elem[0], returns] = (cl - op)/op

    return cluster_returns


def noised_array(y, eta):

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
    epsilon_std_dev = 0.1

    # Calculer la corrélation initiale
    correlation = 0

    x = y.copy()

    z = y.to_numpy()
    z = np.array([item for sublist in z for item in sublist])

    
    # Boucle pour ajuster l'écart-type du bruit jusqu'à ce que la corrélation atteigne eta
    while correlation < eta:
        # Generate a vector of Gaussian noise
        noise = np.random.normal(0, epsilon_std_dev, len(y))

        for i in range(len(y)):
            x.iloc[i, 0] = y.iloc[i, 0] + noise[i]

        w = x.to_numpy()
        w = np.array([item for sublist in w for item in sublist])
        # Calculate the new correlation
        correlation = np.corrcoef(w, z)[0, 1]

        # Adjust the standard deviation of the noise
        epsilon_std_dev += 0.01  

    return x




def markowitz_weights(cluster_return, constituent_weights, df_cleaned, df, lookback_window, eta):

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

    - eta : target correlation that we want to create between y and 
            its perturbated version
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    OUTPUT : returns the markowitz weights of each cluster
    ----------------------------------------------------------------
    '''

    ## on construit la matrice de corrélation associée à ces returns, c'est donc une matrice de corrélation de return de cluster

    cov_matrix = cluster_return.transpose().cov()

    ## on construit le vecteur d'expected return du cluster (252 jours de trading par an, on passe de rendements journaliers à rendements annualisés)
    
    cluster_target_return = cluster_return(constituent_weights=constituent_weights, df_cleaned=df_cleaned, df=df, lookback_window=[lookback_window[1], lookback_window[1]+1])
    
    expected_returns = noised_array(y=cluster_target_return, eta=eta).transpose().to_numpy()
    
    ef = EfficientFrontier(expected_returns=expected_returns, cov_matrix=cov_matrix)
    ef.max_sharpe()

    markowitz_weights = ef.clean_weights()

    return markowitz_weights


def final_weights(markowitz_weights, constituent_weights):

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

    W = []

    for i in range(len(constituent_weights)):

        m_weights = markowitz_weights[constituent_weights[i][0]]

        for elem in constituent_weights[i][1]:

            W.append([elem[0], elem[1] * m_weights])

    return W


def training_phase(lookback_window, df_cleaned, number_of_clusters, sigma, df, eta, clustering_method='SPONGE'):

    '''
    ----------------------------------------------------------------
    GENERAL IDEA : synthetic function that combines all the previous
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    PARAMS : 

    - lookback_window : list of length 2, [start, end] corresponding 
                        to the range of the lookback_window

    - df_cleaned : cleaned pandas dataframe containing the returns 
                   of the stocks

    - number_of_clusters : integer, corresponding to the number of 
                           clusters

    - sigma : float, corresponding to the dispersion in the intra-
              cluster weights

    - df : pandas dataframe containing the raw data 

    - eta : target correlation that we want to create between y and 
            its perturbated version
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    OUTPUT : returns the overall weights of each stocks in our 
             portfolio
    ----------------------------------------------------------------
    '''

    ## ÉTAPE 1 : on obtient la matrice de corrélation des actifs sur la lookback_window
    correlation_matrix_res = correlation_matrix(lookback_window=lookback_window, df_cleaned=df_cleaned)

    ## ÉTAPE 2 : on obtient la composition des clusters et les centroïdes de ceux-ci
    # PROBLÈME DES ARRONDIS
    cluster_composition = cluster_composition_and_centroid(df_cleaned=df_cleaned, correlation_matrix=correlation_matrix_res, number_of_clusters=number_of_clusters, lookback_window=lookback_window, clustering_method=clustering_method)

    ## poids très proches ... ==> dû au fait qu'on regarde sur un trop petit échantillon (30 jours) ? 

    ## ÉTAPE 3 : on obtient les poids constitutifs de chaque actifs au sein d'un même cluster
    constituent_weights_res = constituent_weights(df_cleaned=df_cleaned, cluster_composition=cluster_composition, sigma=sigma, lookback_window=lookback_window)

    ## ÉTAPE 4 : on obtient les rendements de chaque cluster vu comme un actif
    cluster_return_res = cluster_return(constituent_weights=constituent_weights_res, df_cleaned=df_cleaned, df=df, lookback_window=lookback_window) 

    ## ÉTAPE 5 : on obtient les poids de markowitz de chaque cluster
    markowitz_weights_res = markowitz_weights(cluster_return=cluster_return_res, constituent_weights=constituent_weights_res, df_cleaned=df_cleaned, df=df, lookback_window=lookback_window, eta=eta)

    ## ÉTAPE 6 : on remonte aux poids de chaque actif dans l'ensemble
    W = final_weights(markowitz_weights=markowitz_weights_res, constituent_weights=constituent_weights_res)

    W = pd.DataFrame(columns=['ticker', 'weight'], data=W)

    W.set_index('ticker', inplace=True)
    
    return W


def consolidated_W(number_of_repetitions, lookback_window, df_cleaned, number_of_clusters, sigma, df, clustering_method='SPONGE'):

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

    history = []

    for _ in range(number_of_repetitions):
        W = training_phase(lookback_window=lookback_window, df_cleaned=df_cleaned, number_of_clusters=number_of_clusters, sigma=sigma, df=df, clustering_method=clustering_method)
        history.append(W)

    consolidated_W = pd.DataFrame(index=df_cleaned.index, columns=['weight'])

    stock_name = list(df_cleaned.index)

    for i in range(len(stock_name)):

        consolidated_W.loc[stock_name[i], 'weight'] = 0

        for j in range(number_of_repetitions):

            if stock_name[i] in history[j].index:

                consolidated_W.loc[stock_name[i], 'weight'] += history[j].loc[stock_name[i], 'weight']
        
        consolidated_W.loc[stock_name[i], 'weight'] /= number_of_repetitions


    return consolidated_W



def portfolio_returns(evaluation_window, df_cleaned, lookback_window, consolidated_W):

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
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    OUTPUT : returns the portfolio return of each cluster in a 
             pandas dataframe
    ----------------------------------------------------------------
    '''

    evaluation_set = df_cleaned.iloc[:, lookback_window[1]:lookback_window[1]+evaluation_window]

    portfolio_returns = pd.DataFrame(index=evaluation_set.columns, columns=['portfolio return'], data=np.zeros(len(evaluation_set.columns)))

    for elem1 in portfolio_returns.index:
        for stock in list(evaluation_set.index):
            portfolio_returns.loc[str(elem1), 'portfolio return'] += consolidated_W.loc[stock, 'weight']*evaluation_set.loc[stock, str(elem1)]

    return portfolio_returns


def plot_cum_return(overall_return):
    # Tracé du PnL cumulatif
    plt.figure(figsize=(10, 6))
    plt.plot(overall_return.cumsum(), label='Cumulative PnL')
    plt.title('Cumulative Profit and Loss (PnL) of the Portfolio')
    plt.xlabel('Time')
    plt.ylabel('Cumulative PnL')
    plt.legend()
    plt.grid(True)

    plt.xticks(rotation=90)
    plt.xticks(fontsize=6)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.show()


def bar_plot_PnL(PnL):

    '''
    ----------------------------------------------------------------
    GENERAL IDEA : Plot daily PnL using a barplot  
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    PARAMS : 

    - PnL : 
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    OUTPUT : returns the portfolio return of each cluster in a 
             pandas dataframe
    ----------------------------------------------------------------
    '''

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(PnL)), PnL, color='blue', alpha=0.7)

    # Customize the plot
    plt.xlabel('Time Period')
    plt.ylabel('Portfolio Return')
    plt.title('Portfolio Returns Over Time')
    plt.xticks(range(len(PnL)), ['daily_{}'.format(i) for i in range(len(PnL))], rotation=45, ha='right')

    # Show the plot
    plt.tight_layout()
    plt.show()