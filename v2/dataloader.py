
import polars as pl
from polars import selectors as cs
from sklearn.model_selection import ShuffleSplit
import numpy as np

import data

class DataLoader:
    
    MS_NULL_TOL = 0.2
    ORGAN_NULL_TOL = 0.6
    RNG_STATE = 346
    TEST_FRAC = 0.2
    
    def __init__(self, rng_state=None, female_only=False):
        samp_desc = data.read_sampledesc()
        multiorg_olink = data.read_multiorgan_olink()
        #if not female_only:
        #multiorg_olink = data.clear_excess_nullcols(multiorg_olink, self.ORGAN_NULL_TOL)
        serum = data.read_serum()
        serum = data.clear_excess_nullcols(serum, self.MS_NULL_TOL)
        dataset = data.combine(samp_desc, multiorg_olink, serum)
        dataset = dataset.select(cs.exclude(cs.ends_with("_right")))
        if female_only:
            dataset = dataset.filter(pl.col('Sex').eq('F'))
        
        demograph = dataset.select(pl.col('Sex'), pl.col('Strain'))
        
        dataset = dataset.select(cs.numeric())
        
        n = dataset.select(pl.len()).collect().item()
        empty = np.zeros((n, 1))
        if rng_state is None:
            rng_state = self.RNG_STATE
        splitter = ShuffleSplit(test_size=self.TEST_FRAC, random_state=rng_state)
        train_idx, test_idx = next(splitter.split(empty))
        
        train_data = data.get_subset_indices(dataset, train_idx)
        test_data = data.get_subset_indices(dataset, test_idx)
        
        train_demo = data.get_subset_indices(demograph, train_idx)
        test_demo = data.get_subset_indices(demograph, test_idx)
        
        self.__train_data = train_data
        self.__test_data = test_data
        
        self.__train_demographics = train_demo.select(cs.exclude("index"))
        self.__test_demographics = test_demo.select(cs.exclude("index"))
    
    def get_train_predictors(self):
        predictors = data.get_predictors(self.__train_data)
        predictors = predictors.select(pl.exclude("index"))
        
        return predictors.collect()
    
    def get_test_predictors(self):
        predictors = data.get_predictors(self.__test_data)
        predictors = predictors.select(pl.exclude("index"))
        return predictors.collect()
    
    def get_train_target(self, target_name):
        targets = data.get_targets(self.__train_data)
        mytarget = targets.select(pl.col(target_name))
        return mytarget.collect()[target_name]
    
    def get_test_target(self, target_name):
        targets = data.get_targets(self.__test_data)
        mytarget = targets.select(pl.col(target_name))
        return mytarget.collect()[target_name]
    
    def get_train_demographics(self):
        return self.__train_demographics.collect()
    
    def get_test_demographics(self):
        return self.__test_demographics.collect()
    
    @staticmethod
    def get_demographic_dimensions():
        dims = {'Sex': ['F', 'M'], 'Strain': ['B6', 'HET3']}
        return dims


