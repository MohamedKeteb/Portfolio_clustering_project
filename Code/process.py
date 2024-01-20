import numpy as np
import pandas as pd
import ast
import sys
import plotly.graph_objects as go


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

    A_pos_sparse = sparse.csc_matrix(A_pos.values)
    A_neg_sparse = sparse.csc_matrix(A_neg.values)

    data = (A_pos_sparse, A_neg_sparse)

    cluster = Cluster(data)

    ## On applique la méthode SPONGE : clusters the graph using the Signed Positive Over Negative Generalised Eigenproblem (SPONGE) clustering.

    return cluster.SPONGE(k)


def correlation_matrix(lookback_window, df_cleaned):


    '''
    ----------------------------------------------------------------
    GENERAL IDEA : compute the correlation matrix of different stock 
                   returns  over a given lookback_window
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    PARAMS : 
    
    - lookback_window : integer
    
    - df_cleaned : pandas dataframe containing the returns of the stocks

    ----------------------------------------------------------------
    '''
 
    df = df_cleaned.iloc[:, :lookback_window+1] ## + 1 because we don't want to take into account the first column
    df.set_index('ticker', inplace=True)
    correlation_matrix = df.iloc[:, :lookback_window+1].transpose().corr(method='pearson') ## MODIFIÉ
    return correlation_matrix

    ## ==> AVERAGE ON WEIGHTS ?

## we compute the return_centroid of each cluster to attribute intra-cluster weights according to the distance between stocks within the cluster and this 
## centroid

def cluster_composition_and_centroid(df_cleaned, correlation_matrix, number_of_clusters, lookback_window):

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

    - lookback_window : integer, corresponding to the number of lookback 
                        days (in terms of historcal returns)
    ----------------------------------------------------------------
    '''
    ## STEP 1: run the SPONGE clustering algorithm with a number of clusters fixed to be number_of_clusters and using 
    ##         the correlation matrix correlation_matrix ==> we store the results in result

    result = pd.DataFrame(index=list(correlation_matrix.columns), columns=['Cluster label'], data=apply_SPONGE(correlation_matrix, number_of_clusters))

    df_cleaned.set_index('ticker', inplace=True)

    ## STEP 2: compute the composition of each cluster (in terms of stocks)

    cluster_composition = []

    for i in range(1, number_of_clusters):

        if i in result['Cluster label'].values: ## we check that the i-th cluster is not empty

            cluster_composition.append([f'cluster {i}', list(result[result['Cluster label'] == i].index)])

    ## STEP 3: compute the centroid of each cluster 

    for i in range(len(cluster_composition)):

        return_centroid = np.zeros(lookback_window) ## we prepare the return_centroid array to stock the centroid

        for elem in cluster_composition[i][1]:

            return_centroid = return_centroid + df_cleaned.loc[elem, :][:lookback_window].values

        cluster_composition[i].append(return_centroid) ## the third element contains the centroid of the cluster in question

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

    - lookback_window : integer, corresponding to the number of lookback 
                        days (in terms of historcal returns)
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    OUTPUT : modifies in-place the numpy ndarray returned by the 
             cluster_composition_and_centroid function
    ----------------------------------------------------------------
    '''

    constituent_weights = []

    for i in range(len(cluster_composition)):

        weights = []

        for elem in cluster_composition[i][1]:

            elem_returns = df_cleaned.loc[elem, :][:lookback_window].values

            distance_to_centroid = np.linalg.norm(cluster_composition[i][2] - elem_returns)**2

            weights.append([elem, np.exp(-distance_to_centroid/(2*(sigma**2)))])
        
        constituent_weights.append([cluster_composition[i][0], weights])

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

    - lookback_window : integer, corresponding to the number of lookback 
                        days (in terms of historcal returns)
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    OUTPUT : create a single column pandas dataframe containing the 
             return of each cluster over the lookback_window days
    ----------------------------------------------------------------
    '''

    ## len(constituent_weights) =  number of cluster cluster (by construction)
    cluster_return = pd.DataFrame(index=None, columns=[f"cluster {i+1}" for i in range(len(constituent_weights))])

    for i in range(len(constituent_weights)):
        res = 0
        for elem in constituent_weights[i][1]:
            res += elem[1]*df_cleaned.loc[elem[0], :][:lookback_window].values
        
        cluster_return[f"cluster {i+1}"] = res

    return cluster_return

def makowitz_weights(cluster_return, lookback_window):

    '''
    ----------------------------------------------------------------
    GENERAL IDEA : compute the markowitz weights of each cluster in 
                   the synthetic portfolio using the pypfopt package
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    PARAMS : 

    - cluster_return : numpy array as returned by the 
                       cluster_return function 

    - lookback_window : integer, corresponding to the number of lookback 
                        days (in terms of historcal returns)
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    OUTPUT : returns the markowitz weights of each cluster
    ----------------------------------------------------------------
    '''

    ## on construit la matrice de corrélation associée à ces returns, c'est donc une matrice de corrélation de return de cluster

    cov_matrix = cluster_return.cov()

    ## on construit le vecteur d'expected return du cluster (250 jours de trading par an, on passe de rendements journaliers à rendements annualisés)
    expected_returns = (cluster_return.mean(axis=0) + 1)**(250/lookback_window) - 1 ## on fait ici le choix de prendre le rendement moyen comme objectif

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
    OUTPUT : returns the markowitz weights of each cluster
    ----------------------------------------------------------------
    '''

    ### On cherche désormais à calculer le poids de chaque actif dans le portefeuille total

    W = []

    for i in range(len(constituent_weights)):

        m_weights = markowitz_weights[constituent_weights[i][0]]

        for elem in constituent_weights[i][1]:

            W.append([elem[0], elem[1] * m_weights])

    return W


def training_phase(lookback_window, df_cleaned, number_of_clusters):

    ## ÉTAPE 1 : on obtient la matrice de corrélation des actifs sur la lookback_window
    correlation_matrix_res = correlation_matrix(lookback_window=lookback_window, df_cleaned=df_cleaned)

    ## ÉTAPE 2 : on obtient la composition des clusters et les centroïdes de ceux-ci
    # PROBLÈME DES ARRONDIS
    cluster_composition = cluster_composition_and_centroid(df_cleaned=df_cleaned, correlation_matrix=correlation_matrix_res, number_of_clusters=number_of_clusters, lookback_window=lookback_window)

    ## poids très proches ... ==> dû au fait qu'on regarde sur un trop petit échantillon (30 jours) ? 

    ## ÉTAPE 3 : on obtient les poids constitutifs de chaque actifs au sein d'un même cluster
    constituent_weights_res = constituent_weights(df_cleaned=df_cleaned, cluster_composition=cluster_composition, sigma=10, lookback_window=lookback_window)

    ## ÉTAPE 4 : on obtient les rendements de chaque cluster vu comme un actif
    cluster_return_res = cluster_return(constituent_weights=constituent_weights_res, df_cleaned=df_cleaned, lookback_window=lookback_window) 

    ## ÉTAPE 5 : on obtient les poids de markowitz de chaque cluster
    markowitz_weights_res = makowitz_weights(cluster_return=cluster_return_res, lookback_window=lookback_window)

    ## ÉTAPE 6 : on remonte aux poids de chaque actif dans l'ensemble
    W = final_weights(markowitz_weights=markowitz_weights_res, constituent_weights=constituent_weights_res)
    
    return W

def portfolio_return(evaluation_window):
    evaluation_set = df_cleaned.iloc[:, L+1:L+1+evaluation_window]

    evaluation_set = df_cleaned.iloc[:, L+1:L+1+evaluation_window]

    portfolio_return = pd.DataFrame(index=evaluation_set.columns, columns=['portfolio return'], data=np.zeros(len(evaluation_set.columns)))

    for elem1 in portfolio_return.index:
        for elem2 in W:
            portfolio_return.loc[str(elem1), 'portfolio return'] += elem2[1]*evaluation_set.loc[elem2[0], str(elem1)]

    return portfolio_return

def Sharpe_and_PnL(portfolio_return):
    portfolio_return = np.array(portfolio_return)

    # Calcul du rendement moyen du portefeuille
    Rp = np.mean(portfolio_return)

    # Calcul de l'écart type du portefeuille
    sigma_p = np.std(portfolio_return)

    # Taux sans risque (ou rendement moyen du marché)
    Rf = 0.02  # Remplacez par le taux sans risque ou le rendement moyen du marché approprié

    # Calcul du Sharpe ratio
    SR = (Rp - Rf) / sigma_p

    # Calcul des PNL
    PNL = np.cumsum(portfolio_return)

    return SR, PNL


def plot_PnL(SR, PNL):

    # Création du graphique
    fig = go.Figure()

    # Ajout de la trace pour les PNL
    fig.add_trace(go.Scatter(x=np.arange(31, 61), y=PNL, mode='lines', name='PNL'))

    # Ajout d'une ligne pour le Sharpe ratio
    fig.add_shape(
        type='line',
        x0=31, x1=60, y0=0, y1=0,
        line=dict(color='red', width=2),
        opacity=0.7,
        name='Sharpe Ratio'
    )

    # Ajout d'une annotation pour le Sharpe ratio
    fig.add_annotation(
        x=45,
        y=PNL[-1],
        text=f'Sharpe Ratio: {SR:.4f}',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='red',
        font=dict(size=10),
        bordercolor='red',
        borderwidth=2,
        bgcolor='white',
        opacity=0.8
    )

    # Mise en forme du graphique
    fig.update_layout(
        title='Évolution des PNL et Sharpe Ratio',
        xaxis_title='Période',
        yaxis_title='PNL Cumulatif',
        legend=dict(x=0, y=1),
        template='plotly_white'
    )

    # Affichage du graphique
    fig.show()