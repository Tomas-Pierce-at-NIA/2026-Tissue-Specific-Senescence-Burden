# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 12:22:40 2026

@author: piercetf
"""

import pymc as pm
from sklearn import model_selection as model_sel
from sklearn import linear_model as lin
from sklearn import ensemble
from sklearn import impute
from sklearn import pipeline
from sklearn import preprocessing as pre
from sklearn import metrics
from sklearn import dummy
import polars as pl
from polars import selectors as cs
from scipy import stats

import numpy as np
from matplotlib import pyplot
#import numpy as np

# fraction of missing data we are willing to tolerate at most before
# we will not attempt to use that column
MISSINGNESS_TOL = 0.20

# fraction of data held out for testing
TEST_FRAC = 0.20

def load_data() -> (pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame):
    "Load data from disk"
    
    # contains both organ senescence markers and serum olink
    SENESCENCE_FILE = "data/Multiorgan_senescence_and_Olink.csv"
    # contains only serum LC-MS proteomics
    SERUM_FILE = "data/Mouse Serum Samples (280 samples)_Protein_Group_Panel.tsv"
    # maps between organ senescence / olink samples and serum proteomics
    SAMPLE_DESC_FILE = "data/Sample description for Seer.xlsx"
    
    # formatting is scuffed
    sample_desc = pl.read_excel(SAMPLE_DESC_FILE,
                                has_header=False,
                                read_options={"skip_rows":2})
    # formatting is scuffed
    sample_desc.columns = ["Idx", 
                           "Plate ID",
                           "Sample Name",
                           "Sample Type",
                           "Species",
                           "Condition",
                           "Animal ID",
                           "Age (weeks)",
                           "Sex",
                           "Strain",
                           "Age group (3 groups)"]
    
    
    # cursed format convention makes the least painful exercise loading as 
    # strings and doing the conversions ourself
    organ_senescence = pl.read_csv(SENESCENCE_FILE,
                                   missing_utf8_is_empty_string=False, 
                                   infer_schema=False)
    
    # not confident with imputation yet regarding OLINK, so leave as is
    #organ_senescence = organ_senescence.select(pl.all().replace(None, "0.0"))
    organ_senescence = organ_senescence.select(pl.all().replace(pl.lit("FALSE"), None))
    organ_senescence = organ_senescence.select(pl.all().replace(pl.lit("excluded"), None))
    organ_senescence = organ_senescence.select(pl.all().replace(pl.lit("NA"), None))
    organ_senescence = organ_senescence.select(pl.all().replace(pl.lit("needs second IHC"), None))
    organ_senescence = organ_senescence.select(pl.all().replace(pl.lit("no image"), None))
    
    organ_senescence = (organ_senescence
                        .select(
                            pl.col(["Animal Tag", "Strain", "Sex"]),
                            cs.exclude(["Animal Tag", "Strain", "Sex"]).cast(pl.Float64)
                            )
                        )
    
    # split organ senescence from serum OLINK
    
    organs = organ_senescence.select(
        pl.col('Animal Tag'),
        pl.col('Strain'),
        pl.col('Sex'),
        pl.col('Age (weeks)'),
        cs.ends_with('p16'),
        cs.ends_with('p21'),
        cs.ends_with('gH2AX'),
        )
    olink = organ_senescence.select(
        cs.exclude([
            cs.ends_with('p16'),
            cs.ends_with('p21'),
            cs.ends_with('gH2AX')
            ])
        )
    serum_proteome = pl.read_csv(SERUM_FILE, 
                                 separator='\t')
    
    return organs, olink, serum_proteome, sample_desc


def ready_table1(organs :pl.DataFrame, serum_proteome :pl.DataFrame, sample_desc :pl.DataFrame) -> pl.DataFrame:
    """
    organize table suitable for task of predicting organ-specific senescence burden
    from serum LC-MS proteome
    """
    
    serum = serum_proteome.pivot(on="Protein Names",
                                 index="Sample Name",
                                 values="Normalized Intensity (Log10)",
                                 aggregate_function="sum")
    sample_lookup = sample_desc.select(pl.col("Sample Name"),
                                       pl.col("Animal ID")
                                       ).drop_nulls()
    serum_animal = serum.join(sample_lookup,
                              how='inner',
                              on='Sample Name',
                              validate='1:1')
    table1 = organs.join(serum_animal, 
                         left_on="Animal Tag",
                         right_on="Animal ID",
                         how="inner",
                         validate="1:1")
    return table1

def filter_excess_missing(table :pl.DataFrame) -> pl.DataFrame:
    "Create new dataframe with no column exceeding our missingness tolerance"
    missingness = table.null_count() / table.shape[0]
    tolerable_colnames = [col for idx, col in enumerate(missingness.columns) if missingness[0,idx] <= MISSINGNESS_TOL]
    sufficiently_present = table.select(pl.col(tolerable_colnames))
    return sufficiently_present


def build_robust_model():
    """
    builds a linear model using Huber loss to be outlier resistant,
    using elastic net penalty to try and regularize & select vars
    using CV to choose alpha and L1 ratio,
    using KNNImputer to deal with missing data
    and standardizing input variables,
    fitting by SGD algorithm
    """
    imputer = impute.KNNImputer(n_neighbors=5,weights="uniform",keep_empty_features=True)
    #standard = pre.StandardScaler()
    robust_scaler = pre.RobustScaler()
    sgd = lin.SGDRegressor(loss="huber", 
                           penalty="elasticnet", 
                           random_state=1258,
                           max_iter=10_000)
    cv_search = model_sel.RandomizedSearchCV(sgd,
                                             {"alpha": stats.expon(scale=0.2),
                                              "l1_ratio": stats.beta(a=3, b=1),
                                              "epsilon": stats.beta(a=2, b=2)},
                                             refit=True,
                                             random_state=2026_01,
                                             n_iter=100,
                                             n_jobs=10)
    pipe = pipeline.Pipeline([("imputer", imputer),
                              ("robust_scaling", robust_scaler),
                              ("cv_sgd", cv_search)])
    return pipe
    
    #model_sel.RandomizedSearchCV()


def build_enet_model():
    """
    builds an Elastic Net model using CV to choose alpha and L1 ratio,
    using a KNNImputer to deal with missing data
    """
    imputer = impute.KNNImputer(n_neighbors=5,weights="uniform",keep_empty_features=True)
    #standard = pre.StandardScaler()
    robust_scale = pre.RobustScaler()
    enet_cv = lin.ElasticNetCV(l1_ratio=[0.05, 0.1, 0.2, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
                               alphas=10,
                               fit_intercept=True,
                               cv=5,
                               random_state=552+1,
                               max_iter=30_000,
                               n_jobs=10,
                               selection="random")
    pipe = pipeline.Pipeline([("imputer", imputer),
                              ("robust_scaling", robust_scale),
                              ("Elastic Net", enet_cv)])
    return pipe


def build_ard_model():
    """
    builds an ARD model with a KNNImputer to deal with missing data
    """
    imputer = impute.KNNImputer(n_neighbors=5,weights="uniform",keep_empty_features=True)
    #standard = pre.StandardScaler()
    robust_scaler = pre.RobustScaler()
    ard = lin.ARDRegression(max_iter=900)
    pipe = pipeline.Pipeline([("imputer", imputer),
                              ("robust_scaler", robust_scaler),
                              ("ARD", ard)])
    return pipe

def build_hgb_model():
    """
    builds a ensemble tree model (specifically histogram-based gradient boosting)
    has independent means of handling missing data but use imputation anyway
    to also use robust scaling

    """
    imputer = impute.KNNImputer(n_neighbors=5, weights="uniform",keep_empty_features=True)
    robust_scaler = pre.RobustScaler()
    forest = ensemble.HistGradientBoostingRegressor(l2_regularization=0.01,
                                                    max_features=1.0,
                                                    random_state=1206,
                                                    interaction_cst="pairwise")
    pipe = pipeline.Pipeline([("imputer", imputer),
                              ("robust_scaler", robust_scaler),
                              ("histgrad boost", forest)])
    return pipe


def evaluate(model, x_test, y_test, title=""):
    "Evaluate an arbitraty skikit-learn model on OOS data by its MSE, MAE, and R2"
    
    test_pred = model.predict(x_test)
    
    mse = metrics.mean_squared_error(y_test, test_pred)
    mae = metrics.mean_absolute_error(y_test, test_pred)
    r2 = metrics.r2_score(y_test, test_pred)
    
    ped = metrics.PredictionErrorDisplay(y_true=y_test.to_numpy(),
                                         y_pred=test_pred)
    
    ped.plot(kind="residual_vs_predicted")
    pyplot.title(title)
    pyplot.savefig(f"out/residuals_{title}.png")
    pyplot.show()
    ped.plot(kind="actual_vs_predicted")
    pyplot.title(title)
    pyplot.savefig(f"out/actuals_{title}.png")
    pyplot.show()
    
    return {"MSE": mse, "MAE": mae, "R2": r2}


def attempt_multiple_models(x_cols, y_cols):
    """
    Evaluate multiple different model strategies by using this dataset
    split into a test and train split
    """
    
    
    x_train, x_test, y_vars_train, y_vars_test = model_sel.train_test_split(x_cols, 
                               y_cols, 
                               test_size=TEST_FRAC, 
                               random_state=2026_01_22+1)
    
    enet_perf = {}
    enets = {}
    
    ard_perf = {}
    ards = {}
    
    forests_perf = {}
    forests = {}
    
    robust_perf = {}
    robusts = {}
    
    dummy_perf = {}
    dummies = {}
    
    print("ready")
    for idx in range(y_vars_train.shape[1]):
        
        target_name = y_vars_train.columns[idx]
        
        y_train = y_vars_train[:,idx]
        y_test = y_vars_test[:,idx]
        
        no_example = y_train.is_null()
        y_train_local = y_train.filter(~no_example)
        x_train_local = x_train.filter(~no_example)
        
        enet_model = build_enet_model()
        
        ard_model = build_ard_model()
        
        forest_model = build_hgb_model()
        
        robust_model = build_robust_model()
        
        dummy_model = dummy.DummyRegressor(strategy="median")
        
        dummy_model.fit(x_train_local, y_train_local)
        
        enet_model.fit(x_train_local, y_train_local)
        ard_model.fit(x_train_local, y_train_local)
        forest_model.fit(x_train_local, y_train_local)
        robust_model.fit(x_train_local, y_train_local)
        
        no_result = y_test.is_null()
        y_test_local = y_test.filter(~no_result)
        x_test_local = x_test.filter(~no_result)
        
        dummy_oos = evaluate(dummy_model,
                             x_test_local,
                             y_test_local,
                             f"Dummy model {target_name}")
        
        enet_perf_oos = evaluate(enet_model, 
                                 x_test_local, 
                                 y_test_local, 
                                 f"Elastic Net {target_name}")
        
        forest_perf_oos = evaluate(forest_model,
                               x_test_local,
                               y_test_local,
                               f"Hist Grad Boost {target_name}")
        
        robust_perf_oos = evaluate(robust_model,
                                   x_test_local,
                                   y_test_local,
                                   f"Huber SGD ElasticNet {target_name}")
        
        enet_perf[target_name] = enet_perf_oos
        enets[target_name] = enet_model
        
        forests_perf[target_name] = forest_perf_oos
        forests[target_name] = forest_model
        
        robust_perf[target_name] = robust_perf_oos
        robusts[target_name] = robust_model
        
        dummy_perf[target_name] = dummy_oos
        dummies[target_name] = dummy_model
        
        print("*")
    perfs = (enet_perf, forests_perf, robust_perf, dummy_perf)
    models = (enets, forests, robusts, dummies)
    kinds = ("ElasticNet", "HistGradBoost", "Robust", "Dummy")
    return perfs, models, kinds


def display_relative_perf(perfs, kinds, target:str):
    mse = [perfs[i][target]['MSE'] for i in range(len(perfs))]
    x_pos = np.arange(len(mse))
    pyplot.bar(x_pos, mse)
    pyplot.xticks(x_pos, kinds)
    pyplot.title(f"MSE - {target}")
    pyplot.savefig(f"out/multimodel_MSE_{target}.png")
    pyplot.show()
    
    mae = [perfs[i][target]['MAE'] for i in range(len(perfs))]
    x_pos = np.arange(len(mae))
    pyplot.bar(x_pos, mae)
    pyplot.xticks(x_pos, kinds)
    pyplot.title(f"MAE - {target}")
    pyplot.savefig(f"out/multimodel_MAE_{target}.png")
    pyplot.show()
    
    r2 = [perfs[i][target]['R2'] for i in range(len(perfs))]
    x_pos = np.arange(len(r2))
    pyplot.bar(x_pos, r2)
    pyplot.xticks(x_pos, kinds)
    pyplot.title(f"R2 - {target}")
    pyplot.savefig(f"out/multimodel_R2_{target}.png")
    pyplot.show()



if __name__ == '__main__':
    organs, olink, serum_proteome, sample_desc = load_data()
    print("loaded")
    table1 = ready_table1(organs, serum_proteome, sample_desc)
    table1_present = filter_excess_missing(table1)
    table1p_numeric = (table1_present
                       # don't need to track b/c all samples from diff animals - iid
                       .select(cs.exclude(["Animal Tag", "Sample Name"]))
                       # need to enable learners which don't tolerate strings
                       .to_dummies(["Strain", "Sex"])
                       )
    
    y_cols = table1p_numeric.select(cs.ends_with("p16"),
                                    cs.ends_with("p21"),
                                    cs.ends_with("gH2AX"))
    x_cols = table1p_numeric.select(cs.exclude([cs.ends_with("p16"),
                                    cs.ends_with("p21"),
                                    cs.ends_with("gH2AX")]))
    perfs, models, kinds = attempt_multiple_models(x_cols, y_cols)
    for y_colname in y_cols.columns:
        display_relative_perf(perfs, kinds, y_colname)
    