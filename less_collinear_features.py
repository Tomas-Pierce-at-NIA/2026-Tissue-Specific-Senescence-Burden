
import polars as pl
from polars import selectors as cs
from sklearn.model_selection import ShuffleSplit
import numpy as np

from sklearn.impute import SimpleImputer, KNNImputer

import data
# partially inspired by 
# https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from collections import defaultdict
from sklearn import decomposition as decomp

from sklearn import preprocessing as pre

import pymc as pm
import nutpie

class DataLoader:
    
    MS_NULL_TOL = 0.2
    ORGAN_NULL_TOL = 0.6
    RNG_STATE = 346
    TEST_FRAC = 0.2
    
    def __init__(self):
        samp_desc = data.read_sampledesc()
        multiorg_olink = data.read_multiorgan_olink()
        multiorg_olink = data.clear_excess_nullcols(multiorg_olink, self.ORGAN_NULL_TOL)
        serum = data.read_serum()
        serum = data.clear_excess_nullcols(serum, self.MS_NULL_TOL)
        dataset = data.combine(samp_desc, multiorg_olink, serum)
        dataset = dataset.select(cs.exclude(cs.ends_with("_right")))
        dataset = dataset.select(cs.numeric(), 
                                 pl.col('Sex').eq('F').alias('is_F'), 
                                 pl.col('Strain').eq('B6').alias('is_B6')
                                )
        
        n = dataset.select(pl.len()).collect().item()
        empty = np.zeros((n, 1))
        splitter = ShuffleSplit(test_size=self.TEST_FRAC, random_state=self.RNG_STATE)
        train_idx, test_idx = next(splitter.split(empty))
        
        train_data = data.get_subset_indices(dataset, train_idx)
        test_data = data.get_subset_indices(dataset, test_idx)
        
        self.__train_data = train_data
        self.__test_data = test_data
    
    def get_train_predictors(self):
        predictors = data.get_predictors(self.__train_data)
        predictors = predictors.select(pl.exclude("index"))
        
        return predictors.collect()
    
    def get_test_predictors(self):
        predictors = data.get_predictors(self.__test_data)
        return predictors.collect()
    
    def get_train_target(self, target_name):
        targets = data.get_targets(self.__train_data)
        mytarget = targets.select(pl.col(target_name))
        return mytarget.collect()[target_name]
    
    def get_test_target(self, target_name):
        targets = data.get_targets(self.__test_data)
        mytarget = targets.select(pl.col(target_name))
        return mytarget.collect()[target_name]


def cluster_features(dtable, n_clusters=512):
    # see https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
    # compute symmetric correlation
    r_stat = spearmanr(dtable).correlation
    corr = (r_stat + r_stat.T) / 2
    np.fill_diagonal(corr, 1)
    # compute a hierarchical clustering
    dist_mtx = 1 - np.abs(corr)
    dist_link = hierarchy.ward(squareform(dist_mtx))
    # convert hierarchical clustering into a flat clustering
    cluster_ids = hierarchy.fcluster(dist_link, n_clusters, 'maxclust')
    return cluster_ids


def select_feats_by_clustering(dtable, flat_clustering):
    "Select a representative for each cluster"
    _clust_ids, clust_indices = np.unique(flat_clustering, return_index=True)
    return dtable[:, clust_indices] 

class ClusterTransform:
    
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.feature_indices = None
    
    def fit(self, dtable):
        self.feature_indices = cluster_features(dtable, self.n_clusters)


class SelectClusterRep(ClusterTransform):
    
    def __init__(self, n_clusters):
        super(SelectClusterRep, self).__init__(n_clusters)
    
    def transform(self, dtable):
        return select_feats_by_clustering(dtable, self.feature_indices)
    
    def fit_transform(self, dtable):
        self.fit(dtable)
        return self.transform(dtable)


class ClusterDecompositionNMF(ClusterTransform):
    
    def __init__(self, n_clusters):
        super(ClusterDecompositionNMF, self).__init__(n_clusters)
        self.cluster_groups = defaultdict(list)
        self.nmfs = {}
        
    def fit(self, dtable):
        super(ClusterDecompositionNMF, self).fit(dtable)
        for feat_idx, cluster_id in enumerate(self.feature_indices):
            self.cluster_groups[cluster_id].append(feat_idx)
            self.nmfs[cluster_id] = decomp.NMF(n_components=1, 
                                               max_iter=500, 
                                               alpha_W=0.01, 
                                               l1_ratio=1.0,
                                               init="nndsvd").set_output(transform='polars')
        for cluster_id, feat_list in self.cluster_groups.items():
            subtab = dtable[:, feat_list]
            self.nmfs[cluster_id].fit(subtab)
    
    def transform(self, dtable):
        subtables = []
        for cluster_id, feat_list in self.cluster_groups.items():
            subtab = dtable[:, feat_list]
            trans_sub = self.nmfs[cluster_id].transform(subtab)
            trans_sub = trans_sub.rename({'nmf0': 'clust{}'.format(cluster_id)})
            subtables.append(trans_sub)
        return pl.concat(subtables, how='horizontal')
    
    def fit_transform(self, dtable):
        self.fit(dtable)
        return self.transform(dtable)


class ClusterDecompositionSparsePCA(ClusterTransform):
    
    def __init__(self, n_clusters):
        super(ClusterDecompositionSparsePCA, self).__init__(n_clusters)
        self.cluster_groups = defaultdict(list)
        self.pcas = {}
    
    def fit(self, dtable):
        super(ClusterDecompositionSparsePCA, self).fit(dtable)
        for feat_idx, cluster_id in enumerate(self.feature_indices):
            self.cluster_groups[cluster_id].append(feat_idx)
            self.pcas[cluster_id] = decomp.SparsePCA(n_components=1, alpha=0.1, ridge_alpha=0.01, max_iter=1000).set_output(transform='polars')
        for cluster_id, feat_list in self.cluster_groups.items():
            subtab = dtable[:, feat_list]
            self.pcas[cluster_id].fit(subtab)
    
    def transform(self, dtable):
        subtables = []
        for cluster_id, feat_list in self.cluster_groups.items():
            subtab = dtable[:, feat_list]
            trans_sub = self.pcas[cluster_id].transform(subtab)
            trans_sub = trans_sub.rename({'sparsepca0': 'clust{}'.format(cluster_id)})
            subtables.append(trans_sub)
        return pl.concat(subtables, how='horizontal')
    
    def fit_transform(self, dtable):
        self.fit(dtable)
        return self.transform(dtable)


dl = DataLoader()
train_x = dl.get_train_predictors()
simple_imputer = KNNImputer()
#simple_imputer = SimpleImputer(strategy="median")
simple_imputer.set_output(transform='polars')
train_x2 = simple_imputer.fit_transform(train_x)
rep_selector = SelectClusterRep(512)
train_x3 = rep_selector.fit_transform(train_x2.select(pl.exclude("is_F", "is_B6", "Age (weeks)")))
train_x3 = pl.concat([train_x3, train_x2.select(pl.col('Age (weeks)'))],
                     how='horizontal'
                    )
std = pre.StandardScaler().set_output(transform='polars')
train_x4 = std.fit_transform(train_x3)
train_x4 = pl.concat([train_x4, train_x2.select(pl.col("is_F"), pl.col("is_B6"))], how='horizontal')
train_y = dl.get_train_target('SK p21')
train_y2, train_x5 = data.clear_null_response(train_y, train_x4)

d_params = train_x5.shape[1]
exp_rel = 200
train_n = train_x5.shape[0]

deg_free = 3
scale = 5

with pm.Model() as model:
    sigma = pm.HalfNormal('sigma', sigma=2.5)
    tau0 = (exp_rel * sigma) / ((d_params - exp_rel)*np.sqrt(train_n))
    
    gbl_shrink = pm.HalfStudentT('global_shrink', nu=3, sigma=tau0)
    
    c2 = pm.InverseGamma('c2', alpha=deg_free/2, beta=deg_free*(scale**2)/2)
    lcl_shrink = pm.HalfStudentT('lcl_shrink', nu=3, sigma=1, shape=d_params)
    local_shrink = lcl_shrink * pm.math.sqrt(c2 / (c2 + (gbl_shrink**2)*(lcl_shrink**2)))
    
    beta = pm.Normal('beta', mu=0, sigma=1, shape=d_params)
    
    weights = pm.Deterministic('weights', beta * local_shrink * gbl_shrink)
    
    x = pm.Data('x', train_x5)
    ydata = pm.Data('ydata', train_y2.log10())
    
    icpt = pm.Normal('icpt', mu=0, sigma=10)
    
    y = pm.Normal('y', 
                  mu = pm.math.dot(x, weights) + icpt,
                  sigma=sigma,
                  observed=ydata
                 )

compiled_model = nutpie.compile_pymc_model(model, backend='jax', gradient_backend='jax')
trace = nutpie.sample(compiled_model, target_accept=0.99)

