import pandas as pd
import numpy as np 
from sklearn.model_selection import ShuffleSplit
import sys 

np.set_printoptions(precision=10)

# ----------------------------------------------------------------

try:

    from tqdm import tqdm

except ImportError:

    print("PyPortfolioOpt package not found. Installing...")

    try:

        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
        from tqdm import tqdm
        
    except Exception as e:
        print(f"Error installing tqdm package: {e}")
        sys.exit(1)

# ----------------------------------------------------------------
        
class EMA_CV:

    def __init__(self, historical_data, lookback_window, beta, number_of_folds):
        self.historcal_data = historical_data
        self.lookback_window = lookback_window
        self.beta = beta
        self.number_of_folds = number_of_folds
        self.xi = self.eigenvalue_estimator()
        self.EMA_CV = self.EMA()


    ######################### 1. We start by randomizing the auxiliary observation matrix  ̃X from Equation (5) along the time axis #########################
    def auxilary_matrix(self):

        ## 1. We extract the data corresponding to the returns of our assets (columns) during these d days (lines)
        X = self.historcal_data.iloc[self.lookback_window[0]:self.lookback_window[1],:] ## shape days * number of stocks
        days = len(self.lookback_window)
        ## 2. We slightly adjust the matrix of observations to get the auxiliary matrix that puts more weight on recent dates

        # Compute the weight matrix : shape (days, days) (if days = 250, shape (250, 250))
        W = np.sqrt(np.diag(days * (1 - self.beta) * self.beta**(np.arange(self.lookback_window[0], self.lookback_window[1])[::-1]) / (1 - self.beta**days)))  
        X_tilde = pd.DataFrame(index=X.index, columns=X.columns, data=np.dot(W, X))

        ## 3. We randomize the auxiliary matrix of observations according to the time axis
        # Randomized_X = X_tilde.transpose().sample(frac=1, axis=1, random_state=42) ## we transpose X as we want to have daily observations of the whole dataset !

        return X_tilde ## shape (days, 695)
    


    ######################### 2. We then split the (randomized) auxiliary observations into K non-overlapping folds of equal size #########################
    def shuffle_split(self):
        # Initialize ShuffleSplit
        shuffle_split = ShuffleSplit(n_splits=self.number_of_folds, test_size=0.2, random_state=42) 
        # test_size=0.2 : 20% des données pour l'ensemble de test, 80% pour l'ensemble d'entraînement.

        # Create empty list to store splits
        splits = []

        # Perform shuffling and splitting
        for train_index, test_index in shuffle_split.split(self.historcal_data.index):
            train_fold = [self.historcal_data.index[i] for i in train_index]
            test_fold = [self.historcal_data.index[i] for i in test_index]
            splits.append((train_fold, test_fold)) ## attention à cette structure

        return splits
    
    ######################### 3. For each K fold configuration, we estimate the sample eigenvectors from the training set #########################

    def eigen_sample(self, train_fold):
        ## 1. We extract the data corresponding to the returns of our assets (columns) during these d days (lines)
        X = self.historcal_data.loc[train_fold] ## shape days * number of stocks
        days = len(train_fold) 

        # Compute the weight matrix : shape (days, days) (if days = 250, shape (250, 250))
        W = np.sqrt(np.diag(days * (1 - self.beta) * self.beta**(np.arange(days)[::-1]) / (1 - self.beta**days)))  

        # We compute the auxiliary matrix
        X_tilde = np.dot(W, X)

        # We compute the training sample exponential moving average 
        sample_expo_cov = np.dot(X_tilde.T, X_tilde)

        # Calculer les vecteurs et valeurs propres de la matrice de covariance
        _, eigenvectors_train = np.linalg.eigh(sample_expo_cov) ## .eigh and not .eig so that the eigenvalues are real 

        return eigenvectors_train
    

    
    def intra_fold_loss(self, test_fold, sample_eigenvector_i): ## we test the data on this test fold

        ## 1. get the fold cardinality 
        fold_cardinality = len(test_fold) ## 20% of the observations

        ## 2. sample vector of the auxiliary observation matrix from the test fold (inspired from the code above)

        days = len(test_fold)
        X = self.historcal_data.loc[test_fold] ## shape (days, 695)

        ## 2. We slightly adjust the matrix of observations to get the auxiliary matrix that puts more weight on recent dates

        W = np.sqrt(np.diag(days * (1 - self.beta) * self.beta**(np.arange(days)[::-1]) / (1 - self.beta**days)))  # shape (days, days)
        X_tilde = pd.DataFrame(index=X.index, columns=X.columns, data=np.dot(W, X)) # shape (days, 695)

        res = (np.dot(sample_eigenvector_i, X_tilde.T) ** 2)  # shape (, 695) * (695, days) = (, days)
        result = np.sum(res) / fold_cardinality ## sum over days / size of the test sample

        return result
    
    def average_loss(self, splits, index):

        res = 0 ## to stock the overall loss

        for (train_fold, test_fold) in splits:

            ## sur chaque fold, on calcule les sample eigenvectors à partir du training fold correspondant

            sample_eigenvector_i = self.eigen_sample(train_fold=train_fold)[:, index] ## on ne garde que l'eigenvector correspondant au bon index

            ## sur chaque fold, on calcule la perte au sein du fold à partir de l'échantillon de test

            res = res + self.intra_fold_loss(test_fold=test_fold, sample_eigenvector_i=sample_eigenvector_i)

        res = res / len(splits) ## we average by the number of folds (which corresponds to the lengths of the splits)

        return res
    

    def eigenvalue_estimator(self, splits):

        number_of_stocks = len(self.historcal_data.columns) ## COLUMNS HAVE TO BE COMPOSED OF THE STOCKS TICKERS

        xi = np.zeros(number_of_stocks)  # initialisation de x

        for i in tqdm(range(number_of_stocks), desc='Calcul en cours', unit='itération'):
            xi[i] = self.average_loss(splits=splits, index=i)   
                        
        return xi


    def EMA(self):

        days = len(self.lookback_window)
        ## compute the sample exponential moving average correlation matrix
        X = self.historcal_data.iloc[self.lookback_window[0]:self.lookback_window[1],:]
        W = np.sqrt(np.diag(days * (1 - self.beta) * self.beta**(np.arange(self.lookback_window[0], self.lookback_window[1])[::-1]) / (1 - self.beta**days)))  
        X_tilde = np.dot(W, X)  # Produit matriciel de X' et W
        S = np.dot(X_tilde.T, X_tilde)

        ## compute the eigenvectors of S
        _, eigenvectors = np.linalg.eigh(S)

        ## computes the estimator 
        X_tilde = self.auxilary_matrix()
        splits = self.shuffle_split()
        eigenvalue_est = self.eigenvalue_estimator(splits=splits)

        # Initialisation de Sigma avec des zéros
        Sigma = np.zeros((S.shape[0], S.shape[1]), dtype=np.complex128)

        # Parcourir chaque vecteur propre et valeur propre
        for i in range(len(self.historcal_data.columns)):
            xi_dagger = eigenvalue_est[i]  # Conjugue de xi
            ui = eigenvectors[:, i]  # i-ème vecteur propre

            # Calcul du produit externe xi^† * ui * ui^† et addition à Sigma
            Sigma += xi_dagger * np.outer(ui, ui) 

        # Sigma est maintenant la somme des produits xi^† * ui * ui^†
        Sigma = pd.DataFrame(index=self.historcal_data.columns, columns=self.historcal_data.columns, data=np.real(Sigma))

        return Sigma