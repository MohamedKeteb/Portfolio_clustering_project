# Here we put (to not delete them ) all the codes that we did at the fist time


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
    
    n_clusters = len(cluster_composition)

    stock_symbols = list(return_data.index)
    
    micro_portfolio_return = pd.DataFrame(index=cluster_composition, columns=return_data.columns).transpose()
    
    for i in range(n_clusters):
        cluster_return = return_data.loc[cluster_composition[i]] ## get all the tickers in one cluster
        
        coordonnee_tickers = [stock_symbols.index(element) for element in cluster]

        weight_cluster = pd.DataFrame(weights_matrix[coordonnee_tickers])

        micro_portfolio_return[cluster_composition[i]] = cluster_return(cluster, weight_cluster, return_data).transpose()
        
    return micro_portfolio_return.transpose()

