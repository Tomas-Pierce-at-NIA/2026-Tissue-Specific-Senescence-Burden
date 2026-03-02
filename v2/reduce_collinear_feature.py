
from scipy.cluster import hierarchy
from scipy.spatial import distance as spat_dist
from scipy import stats
import numpy as np

def cluster_features(dtable):
    transpose = dtable.to_numpy().T
    dist_mtx = spat_dist.pdist(transpose, 'correlation')
    #sq_dist = sq_dist.squareform(dist_mtx)
    dist_link = hierarchy.centroid(dist_mtx)
    return dist_link


    
def get_cluster_root_indices(dist_link, max_clust=512):
    flat = hierarchy.fcluster(dist_link, t=max_clust, criterion='maxclust')
    leads = hierarchy.leaders(dist_link, flat)
    