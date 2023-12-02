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

    def clean_data(self):

        """
        The columns of the dataframe are the following :

        - "open"
        - "high"
        - "low"
        - "close"
        - "volume"
        - "OPCL"
        - "pvCLCL"
        - "prevAdjClose
        - "SPpvCLCL"
        - "sharesOut"
        - "PERMNO"
        - "SICCD"
        - "PERMCO"
        - "prevRawOpen"
        - "prevRawClose"
        - "prevAdjOpen"

        We first focus only on the three following : 
        - "open"
        - "close"
        - "volume"
        """

        data = self.data[['ticker', 'open', 'close', 'volume']]

        list = data['ticker']
        data = data.drop(data.columns[0], axis=1)
        data.set_index(list)

        ## we now compute the returns 
        
        n = self.data.shape[0]

        df2 = pd.DataFrame(index=data.index, columns=['return', 'volume'])
        df2['volume'] = data['volume'] # volume
        for i in range(n):
            x = np.array(data.iloc[i][0].replace('[', '').replace(']', '').split(', '), dtype=float)## open
            y = np.array(data.iloc[i][1].replace('[', '').replace(']', '').split(', '), dtype=float) ## close
            z = (y - x) / x
            df2.iloc[i, 0] = str(z)

        self.data = df2 


    ## We first define distance matrix  

    def gaussian_adjency(self, sigma):

        def gaussian_distance(x, y, sigma):
            return np.exp(-(np.linalg.norm((x-y))**2)/(2*(sigma**2)))

        n_stocks = self.data.shape[0]
        A = np.zeros((n_stocks,n_stocks))

        for i in range(n_stocks):
            for j in range(n_stocks):
                if i == j: 
                    A[i, j] = 1 

                else: 
                    A[i, j] = gaussian_distance(self.data.iloc[i, 0], self.data.iloc[j, 0], sigma)

        return A
    
    def correlation_adjency(self):

        n_stocks = self.data.shape[0]

        A = np.zeros((n_stocks, n_stocks))

        for i in range(n_stocks):

            for j in range(i, n_stocks):

                if i == j: 
                    A[i, j] = 1

                else: 
                    x = self.data.iloc[i,0]
                    y = self.data.iloc[j, 0]
                    A[i, j] = np.corrcoef(x, y)[0, 1]


        return A
    
    def pearson_corr_volume(self, type='volume'):

        """
        a popular similarity measure in the literature is given by the Pearson correlation
        coefficient that measures linear dependence between variables and takes values in 
        [−1, 1]. By interpreting the correlation matrix as a weighted network whose (signed) 
        edge weights capture the pairwise correlations, we cluster the multivariate time series by
        clustering the underlying signed network
        """
        
        n_stocks = self.data.shape[0]


        if type == 'volume': 

            A = np.zeros((n_stocks, n_stocks))

            for i in range(n_stocks):

                for j in range(i, n_stocks):

                    if i == j: 
                        A[i, j] = 1

                    else: 
                        x = np.array(self.data.iloc[i,1].replace('[', '').replace(']', '').split(', '), dtype=float)
                        y = np.array(self.data.iloc[j,1].replace('[', '').replace(']', '').split(', '), dtype=float)
                        A[i, j] = np.corrcoef(x, y)[0, 1]
                        print('*')

        elif type == 'returns':

            A = np.zeros((n_stocks, n_stocks))

            for i in range(n_stocks):

                for j in range(i, n_stocks):

                    if i == j: 
                        A[i, j] = 1

                    else: 
                        x = np.array(self.data.iloc[i,0].replace('[', '').replace(']', '').split(', '), dtype=float)
                        y = np.array(self.data.iloc[j,0].replace('[', '').replace(']', '').split(', '), dtype=float)
                        A[i, j] = np.corrcoef(x, y)[0, 1]

        
        return A + A.transpose() -  np.eye(n_stocks)
