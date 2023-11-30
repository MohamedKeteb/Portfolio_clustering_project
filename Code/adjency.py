import pandas as pd 
import numpy as np

class Adjency: 

    '''

    This class aims at recovering positive and negative adjency matrix 
    given a dataset [x_1, ..., x_n]

    Parameters: 
    
    data : pandas DataFrame/List [x_1, ..., x_n]
    
    '''
    def __init__(self, data):
        self.data = data

    ## We first define distance matrix  

    def gaussian_adjency(self, sigma):

        def gaussian_distance(x, y, sigma):
            return np.exp(-(np.linalg.norm(np.array([x, y], ord=2))^2)/(2*sigma^2))

        n_stocks = self.data.shape[0]
        A = np.zeros((n_stocks,n_stocks))

        for i in range(n_stocks):
            for j in range(n_stocks):
                if i == j: 
                    A[i, j] = 1 

                else: 
                    A[i, j] = gaussian_distance(self.data.iloc[i, 0], self.data.ioc[j, 0], sigma)

        return A
    