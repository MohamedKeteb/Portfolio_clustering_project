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
        self.EMA_CV = self.EMA_CV()


    ######################### 1. We start by randomizing the auxiliary observation matrix  ̃X from Equation (5) along the time axis #########################
    def auxilary_matrix(self):

        ## 1. We extract the data corresponding to the returns of our assets (columns) during these d days (lines)
        X = self.historcal_data.iloc[self.lookback_window[0]:self.lookback_window[1],:] ## shape days * number of stocks

        ## 2. We slightly adjust the matrix of observations to get the auxiliary matrix that puts more weight on recent dates
        W = np.sqrt(np.diag(len(self.lookback_window) * (1 - self.beta) * self.beta**(np.arange(self.lookback_window[0], self.lookback_window[1])[::-1]) / (1 - self.beta**len(self.lookback_window))))  # Compute the weight matrix
        X_tilde = pd.DataFrame(index=X.index, columns=X.columns, data=np.dot(W, X)).transpose()

        ## 3. We randomize the auxiliary matrix of observations according to the time axis
        # Randomized_X = X_tilde.transpose().sample(frac=1, axis=1, random_state=42) ## we transpose X as we want to have daily observations of the whole dataset !

        return X_tilde
    
    ######################### 2. We then split the (randomized) auxiliary observations into K non-overlapping folds of equal size #########################
    def shuffle_split(self):
        # Initialize ShuffleSplit
        shuffle_split = ShuffleSplit(n_splits=self.number_of_folds, test_size=0.2, random_state=42) 
        # test_size=0.2 : 20% des données pour l'ensemble de test, 80% pour l'ensemble d'entraînement.

        # Create empty list to store splits
        splits = []

        # Perform shuffling and splitting
        for train_index, test_index in shuffle_split.split(self.historcal_data.columns):
            train_fold = [self.historcal_data.columns[i] for i in train_index]
            test_fold = [self.historcal_data.columns[i] for i in test_index]
            splits.append((train_fold, test_fold)) ## attention à cette structure

        return splits
    
    ######################### 3. For each K fold configuration, we estimate the sample eigenvectors from the training set #########################
    def eigen_sample(self, train_fold): ## we train the data on this test fold

        X_tilde_train = self.historcal_data.loc[:, train_fold]

        # Calculer la moyenne de l'ensemble d'entraînement
        mean_train = np.mean(X_tilde_train, axis=1)

        # Centrer les données d'entraînement
        centered_train_data = X_tilde_train.sub(mean_train, axis=0)

        # Calculer la matrice de covariance des données d'entraînement
        cov_matrix_train = np.dot(centered_train_data.T, centered_train_data) ## size number of assets * number of assets

        # Calculer les vecteurs et valeurs propres de la matrice de covariance
        _, eigenvectors_train = np.linalg.eigh(cov_matrix_train) ## .eigh and not .eig so that the eigenvalues are real 

        return eigenvectors_train
    
    def intra_fold_loss(self, test_fold, sample_eigenvector_i): ## we test the data on this test fold

        ## 1. get the fold cardinality 
        fold_cardinality = len(test_fold)

        ## 2. sample vector of the auxiliary observation matrix from the test fold (inspired from the code above)

        days = len(test_fold)
        X = self.historcal_data.loc[:,test_fold].transpose()

        ## 2. We slightly adjust the matrix of observations to get the auxiliary matrix that puts more weight on recent dates

        W = np.sqrt(np.diag(days * (1 - self.beta) * self.beta**(np.arange(days)[::-1]) / (1 - self.beta**days)))  # Compute the weight matrix
        X_tilde = pd.DataFrame(index=X.index, columns=X.columns, data=np.dot(W, X)).transpose()

        res = (np.dot(sample_eigenvector_i, X_tilde) ** 2) / fold_cardinality
        result = np.sum(res)

        return result
    
    def average_loss_i(self, splits, index):

        res = 0 ## to stock the overall loss

        for (train_fold, test_fold) in splits:

            ## sur chaque fold, on calcule les sample eigenvectors à partir du training fold correspondant

            sample_eigenvector_i = self.eigen_sample(train_fold=train_fold)[:, index] ## on ne garde que l'eigenvector correspondant au bon index

            ## sur chaque fold, on calcule la perte au sein du fold à partir de l'échantillon de test

            res = res + self.intra_fold_loss(test_fold=test_fold, sample_eigenvector_i=sample_eigenvector_i)

        res = res / len(splits) ## we average by the number of folds (which corresponds to the lengths of the splits)

        return res

    def eigenvalue_estimator(self):

        data = self.auxilary_matrix()
        
        splits = self.shuffle_split()

        number_of_stocks = data.shape[0]

        xi = np.zeros(number_of_stocks)  # initialisation de x

        for index in tqdm(range(number_of_stocks), desc='Calcul en cours', unit='itération'):
            xi[index] = self.average_loss_i(splits=splits, index=index)   
                        
        return xi

    def EMA_CV(self):
        ## compute the sample exponential moving average correlation matrix
        X = self.historcal_data.iloc[self.lookback_window[0]:self.lookback_window[1],:]
        W = np.diag(len(self.lookback_window) * (1 - self.beta) * self.beta**(np.arange(self.lookback_window[0], self.lookback_window[1])[::-1]) / (1 - self.beta**len(self.lookback_window)))  # Compute the weight matrix, no sqrt as we want the real matrix
        res1 = np.dot(X.T, W)  # Produit matriciel de X' et W
        S = np.dot(res1, X)

        ## compute the eigenvectors of S

        _, eigenvectors = np.linalg.eigh(S)

        ## computes the estimator 
        eigenvalue_estimator = self.eigenvalue_estimator()

        # Tailles des matrices
        num_eigenvalues = eigenvalue_estimator.shape[0]
        num_features = eigenvectors.shape[0]

        # Initialisation de Sigma avec des zéros
        Sigma = np.zeros((num_features, num_features), dtype=np.double)

        # Parcourir chaque vecteur propre et valeur propre
        for i in range(num_eigenvalues):
            xi_dagger = eigenvalue_estimator[i]  # Conjugue de xi
            ui = eigenvectors[:, i]  # i-ème vecteur propre

            # Calcul du produit externe xi^† * ui * ui^† et addition à Sigma
            Sigma += xi_dagger * np.outer(ui, ui) 

        # Sigma est maintenant la somme des produits xi^† * ui * ui^†
        Sigma = pd.DataFrame(index=self.historcal_data.columns, columns=self.historcal_data.columns, data=np.real(Sigma))

        return Sigma