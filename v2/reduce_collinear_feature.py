
from scipy.cluster import hierarchy
from scipy.spatial import distance as spat_dist
from scipy import stats
import numpy as np
import polars as pl

def cluster_features(dtable):
    transpose = np.array(dtable).T
    dist_mtx = spat_dist.pdist(transpose, 'correlation')
    dist_link = hierarchy.centroid(dist_mtx)
    return dist_link

def get_cluster_rep_indices(dist_link, max_clust=512):
    flat = hierarchy.fcluster(dist_link, t=max_clust, criterion='maxclust')
    cluster_ids, cluster_reps = np.unique(flat, return_index=True)
    return cluster_reps
    

class ClusterRepSel:
    
    def __init__(self, max_clust):
        self.max_clust = max_clust
        self.__n_clust = None
        self.__cluster_reps = None
    
    def fit_transform(self, df):
        clusters = cluster_features(df)
        self.__cluster_reps = get_cluster_rep_indices(clusters, self.max_clust)
        self.__n_clust = len(self.__cluster_reps)
        sel_df = df[:, self.__cluster_reps]
        return sel_df
    
    def fit(self, df):
        self.fit_transform(df)
    
    def transform(self, df):
        if self.__cluster_reps is None:
            raise RuntimeError("tried to transform with unfitted cluster-rep selector")
        else:
            sel_df = df[:, self.__cluster_reps]
            return sel_df
    
    def get_feature_names_out(self, feature_in):
        if self.__cluster_reps is None:
            raise RuntimeError("cluster-rep selector is not fitted")
        else:
            feat_name = np.array(feature_in)
            feat_out = feat_name[self.__cluster_reps]
            return feat_out

