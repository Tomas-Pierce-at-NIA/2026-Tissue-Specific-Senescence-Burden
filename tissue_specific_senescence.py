# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 12:22:40 2026

@author: piercetf
"""

import pymc as pm
from sklearn import model_selection as model_sel
from sklearn import linear_model as lin
from sklearn import impute
from sklearn import pipeline
from sklearn import preprocessing as pre
from sklearn import metrics
import polars as pl
from polars import selectors as cs

# fraction of missing data we are willing to tolerate at most before
# we will not attempt to use that column
MISSINGNESS_TOL = 0.10

# fraction of data held out for testing
TEST_FRAC = 0.25

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
    organ_senescence = organ_senescence.select(pl.all().replace(None, "0.0"))
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


def build_enet_model():
    """
    builds an Elastic Net model using CV to choose alpha and L1 ratio,
    using a KNNImputer to deal with missing data
    """
    imputer = impute.KNNImputer(n_neighbors=5,weights="uniform",keep_empty_features=True)
    standard = pre.StandardScaler()
    enet_cv = lin.ElasticNetCV(l1_ratio=[0.1, 0.2, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
                               alphas=10,
                               fit_intercept=True,
                               cv=5,
                               random_state=552,
                               max_iter=30_000,
                               n_jobs=10)
    pipe = pipeline.Pipeline([("imputer", imputer),
                              ("standardizer", standard),
                              ("E.Net", enet_cv)])
    return pipe


def evaluate(model, x_test, y_test):
    "Evaluate an arbitraty skikit-learn model on OOS data by its MSE, MAE, and R2"
    
    test_pred = model.predict(x_test)
    
    mse = metrics.mean_squared_error(y_test, test_pred)
    mae = metrics.mean_absolute_error(y_test, test_pred)
    r2 = metrics.r2_score(y_test, test_pred)
    
    return {"MSE": mse, "MAE": mae, "R2": r2}


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
    
    
    x_train, x_test, y_vars_train, y_vars_test = model_sel.train_test_split(x_cols, 
                               y_cols, 
                               test_size=TEST_FRAC, 
                               random_state=2026_01_22)
    
    enet_perf = {}
    enets = {}
    print("ready")
    for idx in range(y_vars_train.shape[1]):
        
        target_name = y_vars_train.columns[idx]
        
        y_train = y_vars_train[:,idx]
        y_test = y_vars_test[:,idx]
        
        no_example = y_train.is_null()
        y_train_local = y_train.filter(~no_example)
        x_train_local = x_train.filter(~no_example)
        
        enet_model = build_enet_model()
        
        enet_model.fit(x_train_local, y_train_local)
        
        no_result = y_test.is_null()
        y_test_local = y_test.filter(~no_result)
        x_test_local = x_test.filter(~no_result)
        
        perf_oos = evaluate(enet_model, x_test_local, y_test_local)
        
        enet_perf[target_name] = perf_oos
        enets[target_name] = enet_model
        
        print("*")
        
        
        
    