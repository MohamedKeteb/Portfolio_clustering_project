import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

## path Nail : '/Users/khelifanail/Documents/GitHub/Portfolio_clustering_project'
## path Jerome : 'C:/Users/33640/OneDrive/Documents/GitHub/Portfolio_clustering_project'
sys.path.append(r'/Users/khelifanail/Documents/GitHub/Portfolio_clustering_project')  # Ajoute le chemin parent

from signet.cluster import Cluster 
from scipy import sparse
from pypfopt.efficient_frontier import EfficientFrontier

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
 

    correlation_matrix = df_cleaned.iloc[lookback_window[0]:lookback_window[1], :].corr(method='pearson') ## MODIFIÉ

    correlation_matrix = correlation_matrix.fillna(0) ## in case there are NaN values, we replace them with 0 

    return correlation_matrix


## we compute the return_centroid of each cluster to attribute intra-cluster weights according to the distance between stocks within the cluster and this 
## centroid

def cluster_composition_and_centroid(df_cleaned, correlation_matrix, number_of_clusters, lookback_window, clustering_method):

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
    if clustering_method == 'SPONGE':
        result = 1 + pd.DataFrame(index=list(correlation_matrix.columns), columns=['Cluster label'], data=apply_SPONGE(correlation_matrix, number_of_clusters))

    if clustering_method == 'signed_laplacian':
        result = 1 + pd.DataFrame(index=list(correlation_matrix.columns), columns=['Cluster label'], data=apply_signed_laplacian(correlation_matrix, number_of_clusters))

    if clustering_method == 'SPONGE_sym':
        result = 1 + pd.DataFrame(index=list(correlation_matrix.columns), columns=['Cluster label'], data=apply_SPONGE_sym(correlation_matrix, number_of_clusters))


    ## STEP 2: compute the composition of each cluster (in terms of stocks)

    cluster_composition = {}

    for i in range(1, number_of_clusters + 1):
        if i in result['Cluster label'].values:
            tickers = list(result[result['Cluster label'] == i].index)

            return_centroid = np.zeros(lookback_window[1] - lookback_window[0])

            for elem in tickers:
                return_centroid = return_centroid + df_cleaned.loc[:, elem][lookback_window[0]:lookback_window[1]].values

            centroid = return_centroid / len(tickers)

            cluster_composition[f'cluster {i}'] = {'tickers': tickers, 'centroid': centroid}

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

    constituent_weights = {}

    for cluster in cluster_composition.keys():
        weights = []
        total_cluster_weight = 0

        for elem in cluster_composition[cluster]['tickers']:

            elem_returns = df_cleaned.loc[:, elem][lookback_window[0]:lookback_window[1]].values

            ## we compute the distance of the stock to the centroid of the cluster
            distance_to_centroid = np.linalg.norm(cluster_composition[cluster]['centroid'] - elem_returns)**2 

            ## we compute the norm exp(-|x|^2/2*sigma^2)
            total_cluster_weight += np.exp(-distance_to_centroid / (2 * (sigma**2)))

            weights.append(np.exp(-distance_to_centroid / (2 * (sigma**2))))

        normalized_weights = [w / total_cluster_weight for w in weights]
        constituent_weights[cluster] = dict(zip(cluster_composition[cluster]['tickers'], normalized_weights))

    return constituent_weights



def cluster_return(constituent_weights, df_cleaned, lookback_window):

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

    cluster_returns = pd.DataFrame(index = df_cleaned.index[lookback_window[0]:lookback_window[1]], columns= [f'cluster {i}' for i in range(1, len(constituent_weights) + 1)], data = np.zeros((len(df_cleaned.index[lookback_window[0]:lookback_window[1]]), len(constituent_weights))))

    for cluster in constituent_weights.keys():

        for ticker, weight in constituent_weights[cluster].items(): 
            ## we transpose df_cleaned to have columns for each ticker
            cluster_returns[cluster] = cluster_returns[cluster] + df_cleaned[ticker][lookback_window[0]:lookback_window[1]]*weight

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
    epsilon_std_dev = 0.001

    # Calculer la corrélation initiale
    correlation = 1
    
    x = y.copy()
    # Boucle pour ajuster l'écart-type du bruit jusqu'à ce que la corrélation atteigne eta

    while correlation > eta:

        # Generate a vector of Gaussian noise
        noise = np.random.normal(0, epsilon_std_dev, len(y))

        x = noise + y

        correlation = x.corr(y.squeeze())

        # Adjust the standard deviation of the noise
        epsilon_std_dev += 0.0005

    return x




def markowitz_weights(cluster_return_res, constituent_weights, df_cleaned, lookback_window, evaluation_window, eta):

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

    ## on construit la matrice de corrélation associée à ces returns, c'est donc une matrice de corrélation de return de cluster

    cov_matrix = cluster_return_res.cov()

    cov_matrix.fillna(0.)

    ## on construit le vecteur d'expected return du cluster (252 jours de trading par an, on passe de rendements journaliers à rendements annualisés)
    
    cluster_target_return = cluster_return(constituent_weights=constituent_weights, df_cleaned=df_cleaned, lookback_window=[lookback_window[1], lookback_window[1]+evaluation_window]).mean()
    
    expected_returns = noised_array(y=cluster_target_return, eta=eta)
    
    ef = EfficientFrontier(expected_returns=expected_returns, cov_matrix=cov_matrix, weight_bounds=(0, 1))
    
    ef.efficient_return(target_return=expected_returns.mean())

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

    W = {}

    for cluster in constituent_weights.keys(): ## we range across all clusters

        for tickers, weight in constituent_weights[cluster].items(): ## we range across all tickers in each cluster

            W[tickers] = weight*markowitz_weights[cluster]

    return W


def training_phase(lookback_window, df_cleaned, number_of_clusters, sigma, evaluation_window, eta, clustering_method='SPONGE'):

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
    cluster_composition = cluster_composition_and_centroid(df_cleaned=df_cleaned, correlation_matrix=correlation_matrix_res, number_of_clusters=number_of_clusters, lookback_window=lookback_window, clustering_method=clustering_method)

    ## poids très proches ... ==> dû au fait qu'on regarde sur un trop petit échantillon (30 jours) ? 

    ## ÉTAPE 3 : on obtient les poids constitutifs de chaque actifs au sein d'un même cluster
    constituent_weights_res = constituent_weights(df_cleaned=df_cleaned, cluster_composition=cluster_composition, sigma=sigma, lookback_window=lookback_window)

    ## ÉTAPE 4 : on obtient les rendements de chaque cluster vu comme un actif
    cluster_return_result = cluster_return(constituent_weights=constituent_weights_res, df_cleaned=df_cleaned, lookback_window=lookback_window) 

    ## ÉTAPE 5 : on obtient les poids de markowitz de chaque cluster
    markowitz_weights_res = markowitz_weights(cluster_return_res=cluster_return_result, constituent_weights=constituent_weights_res, df_cleaned=df_cleaned, lookback_window=lookback_window, evaluation_window=evaluation_window, eta=eta)

    ## ÉTAPE 6 : on remonte aux poids de chaque actif dans l'ensemble
    W = final_weights(markowitz_weights=markowitz_weights_res, constituent_weights=constituent_weights_res)

    W = pd.DataFrame(list(W.items()), columns=['ticker', 'weights'])
    
    W.set_index('ticker', inplace=True)

    return W



            
def consolidated_W(number_of_repetitions, lookback_window, df_cleaned, number_of_clusters, sigma, evaluation_window, eta, clustering_method='SPONGE'):

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
    for _ in range(number_of_repetitions):
        # Assuming training() returns a DataFrame with 'weights' as the column name
        weights_df = training_phase(lookback_window=lookback_window, df_cleaned=df_cleaned, number_of_clusters=number_of_clusters, sigma=sigma, evaluation_window=evaluation_window, eta=eta, clustering_method=clustering_method)

        # Concatenate the results into columns
        consolidated_W = pd.concat([consolidated_W, weights_df], axis=1)

    # Calculate the average along axis 1
    average_weights = consolidated_W.mean(axis=1)

    # Create a DataFrame with the average weights
    consolidated_W = pd.DataFrame({'weight': average_weights})

    consolidated_W = consolidated_W.transpose()

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

    - df : pandas dataframe containing the raw data
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    OUTPUT : returns the portfolio return of each cluster in a 
             pandas dataframe
    ----------------------------------------------------------------
    '''

    portfolio_returns = pd.DataFrame(index=df_cleaned.iloc[lookback_window[1]:lookback_window[1]+evaluation_window, :].index, columns=['return'], data=np.zeros(len(df_cleaned.iloc[lookback_window[1]:lookback_window[1]+evaluation_window, :].index)))

    for ticker in consolidated_W.columns: 

    ##  each time we add :            the present value of the return + the weighted "contribution" of the stock 'ticker' times is weight in the portfolio
        portfolio_returns['return'] = portfolio_returns['return'] + df_cleaned.loc[ticker][lookback_window[1]:lookback_window[1]+evaluation_window]*consolidated_W[ticker]['weight']

    return portfolio_returns


def sliding_window(df_cleaned, lookback_window_0, number_of_clusters, sigma, clustering_method, number_of_repetition, number_of_window, evaluation_window, eta):
    
    PnL = []
    daily_PnL = []
    overall_return = pd.DataFrame()
    portfolio_value=[1] #we start with a value of 1, the list contain : the porfolio value at the start of each evaluation period
    lookback_window = lookback_window_0

    for i in range(1, number_of_window + 1):

        consolidated_W_res = consolidated_W(number_of_repetitions=number_of_repetition, lookback_window=lookback_window, df_cleaned=df_cleaned, number_of_clusters=number_of_clusters, sigma=sigma, evaluation_window=evaluation_window, eta=eta, clustering_method=clustering_method)

        portfolio_return = portfolio_returns(evaluation_window=evaluation_window, df_cleaned=df_cleaned, lookback_window=lookback_window, consolidated_W=consolidated_W_res)

        overall_return = pd.concat([overall_return, portfolio_return])

        lookback_window = [lookback_window_0[0] + evaluation_window*i, lookback_window_0[1] + evaluation_window*i]

        PnL = np.concatenate((PnL, np.reshape(np.cumprod(1 + portfolio_return)*portfolio_value[-1] - portfolio_value[-1], (evaluation_window,))))## car on réinvestit immédiatement après
        daily_PnL = np.concatenate((daily_PnL, np.reshape(np.cumprod(1 + portfolio_return)*portfolio_value[-1] - portfolio_value[-1], (evaluation_window,))))## car on réinvestit immédiatement après

        portfolio_value.append(portfolio_value[-1]+PnL[-1])

        print(portfolio_value[-1])
        
        print(f'step {i}')

    n = len(PnL)//evaluation_window

    for j in range(1, n):

        for i in range(1, evaluation_window+1):
            
            PnL[j*evaluation_window + i - 1] = PnL[j*evaluation_window + i - 1] + PnL[j*evaluation_window - 1]
    
    return overall_return, PnL, portfolio_value, daily_PnL


def save_to_csv(year, clustering_method, daily_PnL, PnL, overall_return):

    '''
    ----------------------------------------------------------------
    GENERAL IDEA : save the outputs of sliding_window() to csv file. 
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    PARAMS : 

    - year : string, corresponding to the year of trading we consider

    - clustering_method : string, corresponding to the name of the 
                          clustering method we use ('SPONGE', 
                          'Signed Laplacian').

    - daily_PnL, PnL, overall_return : outputs of sliding_window()
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    OUTPUT : None
    ----------------------------------------------------------------
    '''


    df_daily = pd.DataFrame(daily_PnL, columns=['Daily PnL'])

    df_daily.to_csv(f'daily_{year}_{clustering_method}.csv', index=False)

    df_PnL = pd.DataFrame(PnL, columns=['PnL'])

    df_PnL.to_csv(f'PnL_{year}_{clustering_method}.csv', index=False)

    df_overall_return = pd.DataFrame(overall_return.values, columns=['Return'])

    df_overall_return.to_csv(f'Overall_return_{year}_{clustering_method}.csv', index=False)


def get_sp500_PnL(start_date, end_date):

    '''
    ----------------------------------------------------------------
    GENERAL IDEA : get the S&P500 index daily PnL between the star
                   and end dates 
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    PARAMS : 

    - start_date, end_date : strings, corresponding to start and end
                             dates. The format is the datetime format
                             "YYYY-MM-DD"

    ----------------------------------------------------------------

    ----------------------------------------------------------------
    OUTPUT : pandas.DataFrame containing the S&P500 index daily 
             between the star and end dates
    ----------------------------------------------------------------
    '''

    # Specify the ticker symbol for S&P 500
    ticker_symbol = "^GSPC"

    # Fetch historical data
    sp500_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    sp500_data['Daily PnL'] = (sp500_data['Close'] - sp500_data['Open']) / sp500_data['Open'][0] ## /100 because we initially invest 1 dollar in our portfolio?
    sp500_PnL = sp500_data['Daily PnL'].transpose() ## we remove the -2 values to have matching values

    return sp500_PnL


def plot_cumulative_PnL(PnL):
    
    # Création de l'axe des abscisses (nombre de jours)
    days = np.arange(1, len(PnL) + 1)

    # Configuration de seaborn pour un style agréable
    sns.set(style="whitegrid")

    # Tracer la PnL cumulative avec seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=days, y=PnL, label='Cumulative PnL', color='blue')

    # Ajouter des titres et des légendes
    plt.title('Cumulative PnL of Portfolio')
    plt.xlabel('Days')
    plt.ylabel('Cumulative PnL')

    # Personnaliser l'axe des ordonnées avec un pas de 0.01
    plt.yticks(np.arange(0, max(PnL) + 0.01, 0.01))

    # Afficher le graphique
    plt.show()

def bar_plot_daily_PnL(daily_PnL):

    '''
    ----------------------------------------------------------------
    GENERAL IDEA : Plot daily PnL using a barplot  
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    PARAMS : 

    - daily_PnL : 
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    OUTPUT : returns the portfolio return of each cluster in a 
             pandas dataframe
    ----------------------------------------------------------------
    '''

    # Création de l'axe des abscisses (nombre de jours)
    days = np.arange(1, len(daily_PnL) + 1)

    # Configuration de seaborn pour un style agréable
    sns.set(style="whitegrid")

    # Tracer l'évolution quotidienne de la PnL sous forme de diagramme à barres avec seaborn
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=days, y=daily_PnL, color='blue', width=0.8)  # Ajustez la largeur ici

    # Rotation des étiquettes de l'axe des abscisses de 45 degrés avec un ajustement
    ax.set_xticks(np.arange(0,251,10))
    ax.set_xticklabels(ax.get_xticks(), rotation=90, ha='right', rotation_mode='anchor')

    # Ajouter des titres et des légendes
    plt.title('Daily PnL Evolution')
    plt.xlabel('Days')
    plt.ylabel('Daily PnL')

    # Afficher le graphique
    plt.show()