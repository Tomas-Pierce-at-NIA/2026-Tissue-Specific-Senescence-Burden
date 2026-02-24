
from sklearn import preprocessing as pre
from sklearn.experimental import enable_iterative_imputer
from sklearn import impute

class DataPrep:
    
    def __init__(self):
        
        self.scaler = pre.RobustScaler()
        self.scaler.set_output(transform='polars')
        
        self.imputer = impute.KNNImputer(n_neighbors=7, weights='distance')
        self.imputer.set_output(transform='polars')
    
    def fit_transform(self, X, y=None):
        scaled = self.scaler.fit_transform(X)
        imputed = self.imputer.fit_transform(scaled)
        return imputed
    
    def fit(self, X, y=None):
        self.fit_transform(X, y)
    
    def transform(self, X):
        scaled = self.scaler.transform(X)
        imputed = self.imputer.transform(scaled)
        return imputed