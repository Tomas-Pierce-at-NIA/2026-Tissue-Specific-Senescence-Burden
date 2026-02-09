
import data

import polars as pl
from polars import selectors as cs

from sklearn import decomposition
from sklearn import preprocessing
from sklearn.impute import KNNImputer

from sklearn.linear_model import ElasticNetCV
from sklearn import cross_decomposition
from sklearn import metrics

VIEW = False

class DataProcessor:
    
    def __init__(self, max_components, rng_seed=429):
        
        self.imputer = KNNImputer()
        self.imputer.set_output(transform='polars')
        
        #self.decomp = decomposition.MiniBatchSparsePCA(n_components=max_components,n_jobs=-1)
        self.decomp = decomposition.MiniBatchNMF(n_components=max_components,
                                                 init='nndsvd',
                                                 alpha_W=0.01,
                                                 l1_ratio=0.2,
                                                 max_iter=10_000,
                                                 forget_factor=1,
                                                 batch_size=32)
        self.decomp.set_output(transform='polars')
        
        self.std = preprocessing.StandardScaler()
        self.std.set_output(transform='polars')
    
    def fit_transform(self, X, y=None):
        imputed = self.imputer.fit_transform(X)
        decomposed = self.decomp.fit_transform(imputed)
        standardized = self.std.fit_transform(decomposed)
        return standardized
    
    def fit(self, X, y=None):
        self.fit_transform(X)
    
    def transform(self, X):
        imputed = self.imputer.transform(X)
        decomposed = self.decomp.transform(imputed)
        standardized = self.std.transform(decomposed)
        return standardized


def prepare_xdata(x_data, data_processor, *_, is_train):
    x_num = x_data.select(cs.numeric())
    if is_train:
        prepped_num = data_processor.fit_transform(x_num)
    else:
        prepped_num = data_processor.transform(x_num)
    x_cat = x_data.select(pl.col('Sex'), pl.col('Strain'))
    prepped_x = pl.concat([prepped_num, x_cat], how='horizontal')
    dummied = prepped_x.to_dummies(['Sex', 'Strain'])
    return dummied

def clear_null_resp(target, predictors):
    valid_mask = ~target.is_null()
    valid_target = target.filter(valid_mask)
    corresp_predictors = predictors.filter(valid_mask)
    return valid_target, corresp_predictors


if __name__ == '__main__':
    x_train, x_test, y_trains, y_tests = data.prepare_datasets(fem_only=True)
    if VIEW:
        from matplotlib import pyplot
        pyplot.hist(y_trains['OV p16'])
        pyplot.title("Distribution of Ovary p16 (train set)")
        pyplot.show()
        #pyplot.hist(y_tests['OV p16'])
        #pyplot.title("Distribution of Ovary p16 (test set)")
        #pyplot.show()
        pyplot.hist(y_trains['OV p16'].log10())
        pyplot.title("Distribution of Log10 Ovary p16 (train set)")
        pyplot.show()
        #pyplot.hist(y_tests['OV p16'].log10())
        #pyplot.title("Distribution of Log10 Ovary p16 (test set)")
        #pyplot.show()
    
    y_train2, x_train2 = clear_null_resp(y_trains['SK p16'], x_train)
    y_test2, x_test2 = clear_null_resp(y_tests['SK p16'], x_test)
    data_proc = DataProcessor(100)
    x_train3 = prepare_xdata(x_train2, data_proc, is_train=True)
    x_test3 = prepare_xdata(x_test2, data_proc, is_train=False)
    
    #x_train2 = prepare_xdata(x_train, data_proc, is_train=True)
    #x_test2 = prepare_xdata(x_test, data_proc, is_train=False)
    #y_train_skp16, x_train_skp16 = clear_null_resp(y_trains['SK p16'], x_train2)
    #y_test_skp16, x_test_skp16 = clear_null_resp(y_tests['SK p16'], x_test2)
    