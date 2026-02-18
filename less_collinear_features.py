
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
import arviz as az

from matplotlib import pyplot
from sklearn import metrics
from matplotlib import patches
from matplotlib import lines as mlines


class DataLoader:
    
    MS_NULL_TOL = 0.2
    ORGAN_NULL_TOL = 0.6
    RNG_STATE = 346
    TEST_FRAC = 0.2
    
    def __init__(self, rng_state=None):
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
        if rng_state is None:
            rng_state = self.RNG_STATE
        splitter = ShuffleSplit(test_size=self.TEST_FRAC, random_state=rng_state)
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


def build_model(train_x, train_y, exp_rel, deg_free=3, scale=5):
    """builds modified hierarchical horseshoe prior regression using input
    training data.
    train_x - training predictors
    train_y - training target
    exp_rel - number of initially expected relevant predictors
    
    following control expected distribution(s) of non-shrunk predictors
    deg_free - degrees of freedom 
    scale - scale parameter
    
    number of training examples and number of training predictors inferred from train_x
    
    constraint:
    exp_rel must be strictly less than number of predictor columns in train_x
    
    Inspirations
    https://mellorjc.github.io/HorseshoePriorswithpymc3.html
    https://arxiv.org/abs/1610.05559
    https://austinrochford.com/posts/2021-05-29-horseshoe-pymc3.html#fn2
    https://projecteuclid.org/journalArticle/Download?urlId=10.1214%2F17-EJS1337SI
    https://github.com/to-mi/stan-survival-shrinkage
    """
    
    train_n, d_params = train_x.shape
    
    # smallest degrees of freedom that consistently converges
    # during sampling
    global_df = 3
    local_df = 3
    
    with pm.Model() as model:
        x = pm.Data('x', train_x)
        ydata = pm.Data('ydata', train_y)
        
        sigma = pm.HalfNormal('sigma', sigma=2.5)
        
        tau0 = (exp_rel * sigma) / ((d_params - exp_rel)*np.sqrt(train_n))
        global_shrink = pm.HalfStudentT('global_shrink',nu=global_df,sigma=tau0)
        
        c2 = pm.InverseGamma('c2', alpha=deg_free/2, beta=deg_free*(scale**2)/2)
        lcl_shrink = pm.HalfStudentT('lcl_shrink',nu=local_df,sigma=1,shape=d_params)
        local_shrink = lcl_shrink * pm.math.sqrt(c2 / (c2 + (global_shrink**2)*(lcl_shrink**2)))
        
        beta = pm.Normal('beta', mu=0, sigma=1, shape=d_params)
        weights = pm.Deterministic('weights', beta * local_shrink * global_shrink)
        
        icpt = pm.Normal('icpt', mu=0, sigma=10)
        
        y=pm.Normal('y', mu=pm.math.dot(x, weights) + icpt, sigma=sigma, observed=ydata)
        
    return model



if __name__ == '__main__':
    dl = DataLoader()
    train_x = dl.get_train_predictors()
    #simple_imputer = KNNImputer()
    simple_imputer = SimpleImputer(strategy="median")

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
    train_y = dl.get_train_target('SK gH2AX')
    train_y2, train_x5 = data.clear_null_response(train_y, train_x4)


    test_x = dl.get_test_predictors()
    test_x2 = simple_imputer.transform(test_x)
    test_x3 = rep_selector.transform(test_x2.select(pl.exclude("is_F", "is_B6", "Age (weeks)")))
    test_x3 = pl.concat([test_x3, test_x2.select(pl.col('Age (weeks)'))], how='horizontal')
    test_x4 = std.transform(test_x3)
    test_x4 = pl.concat([test_x4, test_x2.select(pl.col("is_F"), pl.col("is_B6"))], how="horizontal")
    test_y = dl.get_test_target("SK gH2AX")
    test_y2, test_x5 = data.clear_null_response(test_y, test_x4)
    
    # 262 genes identified as senescence-associated by literature data-mining
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC3273898/
    model = build_model(train_x5, train_y2, 262)
    compiled_model = nutpie.compile_pymc_model(model, backend='jax', gradient_backend='jax')
    #adapting_model = compiled_model.with_transform_adapt()
    trace = nutpie.sample(compiled_model, target_accept=0.99, tune=1000, draws=1000, chains=6, cores=6)
    
    g = model.to_graphviz()
    g.render("modelskel.gv.s", directory="bayes_figs")

    with model:
        prior = pm.sample_prior_predictive()
        trace.extend(prior)
        ppc = pm.sample_posterior_predictive(trace)
        trace.extend(ppc)
        ll = pm.compute_log_likelihood(trace)
        trace.extend(ll)

    with model:
        pm.set_data({'x': test_x5, 'ydata': test_y2})
        oos = pm.sample_posterior_predictive(trace, predictions=True)
        trace.extend(oos)
    
    r2 = az.r2_score(y_true=trace.predictions_constant_data['ydata'].values,
                y_pred=trace.predictions.stack(sample=('chain','draw'))['y'].values.T
                )
    
    print(r2)
    
    az.plot_loo_pit(trace, 'y')
    pyplot.show()
    
    ax = az.plot_ppc(trace)
    ax.legend(loc='upper right')
    pyplot.show()
    
    wsum = az.summary(trace, var_names=['weights'])
    wsum = pl.from_pandas(wsum).with_row_index()
    
    bayes_factors = []
    for idx in wsum['index']:
        bf_dict = az.bayes_factor(trace.isel(weights_dim_0=idx), 'weights', 0.0)
        bf = bf_dict['BF10']
        bayes_factors.append(bf)
    wsum = wsum.with_columns(pl.Series("BayesFactor10", bayes_factors))
    names = pl.Series("PredictorName", train_x5.columns)
    wsum = wsum.with_columns(names)
    
    #hue = wsum.select((pl.col('mean').lt(0) & pl.col('BayesFactor10').gt(1))
    wsum_plot = wsum.with_columns(
        pl.when(pl.col('mean').lt(0) & pl.col('BayesFactor10').gt(10))
        .then(pl.lit(-1))
        .otherwise(
            pl.when(pl.col('mean').gt(0) & pl.col('BayesFactor10').gt(10))
            .then(pl.lit(+1))
            .otherwise(
                pl.when(pl.col('mean').lt(0) & pl.col('BayesFactor10').lt(10))
                .then(pl.lit(-0.1))
                .otherwise(
                    pl.when(pl.col('mean').gt(0) & pl.col('BayesFactor10').lt(10))
                    .then(pl.lit(0.1))
                    .otherwise(pl.lit(0.0))
                )
            )
        ).alias("association_cat")
    )
    
    top = wsum_plot.filter(pl.col('BayesFactor10').log10().gt(1)).sort(pl.col('mean')).tail(7)
    bottom = wsum_plot.filter(pl.col('BayesFactor10').log10().gt(1)).sort(pl.col('mean')).head(5)
    
    lookup = data.gene_lookup_by_name().collect()
    top = top.join(lookup, left_on='PredictorName', right_on='Protein Names', how='left')
    bottom = bottom.join(lookup, left_on='PredictorName', right_on='Protein Names', how='left')
    
    pyplot.scatter(x=wsum['mean'], 
                   y=wsum['BayesFactor10'].log10(), 
                   marker='+', 
                   # easier to fix here
                   c=wsum_plot['association_cat'],
                   cmap="coolwarm")
      
    for i in range(len(top)):
        top_x, top_ye, top_lbl, top_gene = top[i, ['mean', 'BayesFactor10', 'PredictorName', 'Gene Names']].row()
        top_y = np.log10(top_ye)
        if top_gene is not None:
            top_lbl = top_gene.split(';')[0]
        #top_lbl = top_lbl.split(';')[0]
        pyplot.text(top_x + 0.01, top_y + 0.01, top_lbl)
        
    for i in range(len(bottom)):
        bottom_x, bottom_ye, bottom_lbl, bottom_gene = bottom[i, ['mean', 'BayesFactor10', 'PredictorName', 'Gene Names']].row()
        bottom_y = np.log10(bottom_ye)
        if bottom_gene is not None:
            bottom_lbl = bottom_gene.split(';')[0]
        #bottom_lbl = bottom_lbl.split(';')[0]
        kwargs = {'ha':'right'}# if len(bottom_lbl) >= 6 else {'ha':'left'}
        kwargs['va'] = 'top'
        pyplot.text(bottom_x - 0.01, bottom_y - 0.01, bottom_lbl, **kwargs)
        ha_idx += 1
        va_idx+=1
    #pyplot.legend()
    cmap = pyplot.get_cmap('coolwarm')
    #colors = cmap([1.0, 0.1, 0.0, -0.1, -1.0])
    # manually redo color normalization for legend
    colors = cmap([0.0, 0.45, 0.5, 0.55,  1.0])
    minus1 = mlines.Line2D([], [], linestyle='', marker='+', color=colors[0], label='negative\nassociation')
    minus01 = mlines.Line2D([], [], linestyle='', marker='+', color=colors[1], label='limited\nevidence')
    plus01 = mlines.Line2D([], [], linestyle='', marker='+', color=colors[3], label='limited\nevidence')
    plus1 = mlines.Line2D([], [], linestyle='', marker='+', color=colors[4], label='positive\nassociation')
    pyplot.legend(handles=[minus1, minus01, plus01, plus1], loc='lower left')
    
    left, right = pyplot.xlim()
    max_mag = max(abs(left), abs(right))
    pyplot.xlim(-max_mag, max_mag)
    pyplot.ylim(0.0, 2.5)
    pyplot.xlabel("Posterior Coefficient Mean")
    pyplot.ylabel("log10(Bayes Factor)")
    pyplot.axhline(y=0.5, linestyle='--', color='purple')
    pyplot.axhline(y=1.0, linestyle='--', color='purple')
    pyplot.axhline(y=1.5, linestyle='--', color='purple')
    pyplot.axhline(y=2.0, linestyle='--', color='purple')
    
    pyplot.text(x=2.0, y=0.25, s="Anecdotal\nEvidence", color='purple')
    pyplot.text(x=2.0, y=0.75, s="Moderate\nEvidence", color="purple")
    pyplot.text(x=2.0, y=1.25, s="Strong\nEvidence", color="purple")
    pyplot.text(x=2.0, y=1.75, s="Very Strong\nEvidence", color="purple")
    pyplot.text(x=2.0, y=2.25, s="Decisive\nEvidence", color="purple")
    
    pyplot.title("Skin γH2AX predictor associations")
    
    pyplot.show()
