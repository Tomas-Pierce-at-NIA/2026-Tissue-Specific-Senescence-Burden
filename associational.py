

import nutpie
import pymc as pm
import arviz as az
from matplotlib import pyplot
import polars as pl
import numpy as np
from polars import selectors as cs
from sklearn import model_selection as model_select
from sklearn.impute import KNNImputer

SERUM = "data/Mouse Serum Samples (280 samples)_Protein_Group_Panel.tsv"
MULTIORGAN = "data/Multiorgan_senescence_and_Olink.xlsx"
SAMPLE_DESC = "data/Sample description for Seer.xlsx"

# maximum fraction of nullness we are willing to tolerate
# before we will not even attempt interpolation / imputation
NULL_TOL = 0.05

def read_sampledesc():
    table = pl.read_excel(
        SAMPLE_DESC,
        read_options={'header_row':None,
                      'skip_rows':2},
        columns=list(range(1,11))
    )
    table = table.rename({
      "column_1" : "Plate ID",
      "column_2" : "Sample Name",
      "column_3" : "Sample Type",
      "column_4" : "Species",
      "column_5" : "Condition",
      "column_6" : "Animal Tag",
      "column_7" : "Age (weeks)",
      "column_8" : "Sex",
      "column_9" : "Strain",
      "column_10": "Age group"
    })
    return table

# the files are combined this way, not my fault
def read_multiorgan_olink():
    table = pl.read_excel(
         MULTIORGAN,
         schema_overrides = {
             "Animal Tag": pl.String,
             "SK p16": pl.Float64,
             "SK p21": pl.Float64,
             "SK gH2AX": pl.Float64,
             "LIV p16": pl.Float64,
             #"LIV p21": pl.Float64,
             "LIV gH2AX": pl.Float64,
             "SCO p16": pl.Float64,
             "SCO p21": pl.Float64,
             "SCO gH2AX": pl.Float64,
             "OV p16": pl.Float64,
             "OV p21": pl.Float64,
             "OV gH2AX": pl.Float64,
             "Cxcl1": pl.Float64,
         }
    )
    return table

def read_serum():
    lazy = pl.scan_csv(SERUM,
                       separator="\t"
                       )
    prot_list = lazy.select(pl.col("Protein Names").unique()).collect()["Protein Names"]
    
    pivot = lazy.pivot(
        on=pl.col("Protein Names"),
        on_columns=prot_list,
        index=pl.col("Sample Name"),
        values=pl.col("Intensity (Log10)"),
        aggregate_function="sum"
    )
    return pivot

def combine(samp_desc, multi_olink, serum):
    combo1 = multi_olink.join(
        samp_desc,
        on=["Animal Tag", "Age (weeks)"],
        how="inner",
        validate="1:1"
    ).lazy()
    table = serum.join(
        combo1,
        on=["Sample Name"],
        how="inner",
        validate="1:1"
    )
    return table

def load_alldata():
    serum = read_serum()
    multi_olink = read_multiorgan_olink()
    samps = read_sampledesc()
    combo = combine(samps, multi_olink, serum)
    no_dupcols = combo.select(
        cs.exclude(cs.ends_with("_right"))
    )
    return no_dupcols

def collect_targets(alldata):
    return alldata.select(
        cs.ends_with("p16"),
        cs.ends_with("p21"),
        cs.ends_with("gH2AX")
    ).collect()

def collect_predictors(alldata):
    return alldata.select(
        cs.exclude(
            cs.ends_with("p16"),
            cs.ends_with("p21"),
            cs.ends_with("gH2AX")
        )
    ).select(cs.numeric(), 
             pl.col("Sex"), 
             pl.col("Strain")
    ).collect()

class PolarsStandardizer:
    """
    Analog to sckikit-learn's standardizer,
    but specialized for use with polars dataframes
    and specific items potentially unique to our data.
    """
    
    def __init__(self, train_data):
        self.train_means = train_data.select(
            cs.numeric().mean()
        )
        self.train_std = train_data.select(
            cs.numeric().std()
        )
    
    def standardize_numeric(self, data):
        standardized_numeric = data.select(
            [((pl.col(c) - self.train_means[0,c])
               / self.train_std[0,c]
             ) for c in self.train_means.columns
            ]
        )
        non_numeric = data.select(
            cs.exclude(
                cs.numeric()
            )
        )
        standardized_data = pl.concat(
            [standardized_numeric, non_numeric],
            how="horizontal"
        )
        return standardized_data

def clear_null_resp(target, predictors):
    """
    get rid of rows where the target is null
    because the tools aren't set up to deal with that
    """
    valid_mask = ~target.is_null()
    valid_target = target.filter(valid_mask)
    corresp_predictors = predictors.filter(valid_mask)
    return valid_target, corresp_predictors


def undo_inferred_zero(dataframe):
    """
    This merits some more explanation.
    To go from very tall format of having a protein name column
    to having a column for each MS-identified protein, and a row per sample,
    we have to conduct a pivot operation.
    When we conduct a pivot operation, we sometimes see quantications
    for the same protein in multiple rows in originating file.
    To handle this, we sum all occurrences of the protein so that it becomes
    a single quantification of that protein for that sample.
    Because we do this sum operation, and there is an implicit starting point of 
    zero (additive identity) to the sum operation, and we need it to achieve the pivot,
    the  pivot operation will implicitly impute zero (0) values for proteins which are
    present in the data but not observed for that sample.
    Zero measurements of any protein do not occur in MS data, and even if they did,
    they would not be reliable, these methods do not give evidence of absence.
    Direct inspection of the underlying data will also confirm that zero (0) values
    do not occur in either normalized or non-normalized protein measurements,
    and the probability of either of these summing to zero is nill in the direct case
    and almost nill in the normalized case.
    For this reason, we have to replace zero values in our predictors with an indication
    that this is actually a missing value
    """
    return dataframe.select(pl.all().replace(old=0.0, new=None))


def clear_excess_nullcols(dataframe, tol=NULL_TOL):
    """
    remove columns with excessive missing data
    """
    null_count = dataframe.null_count()
    null_frac = null_count / len(dataframe)
    okay_masking = null_frac < tol
    okay = dataframe.select([pl.col(c) for c in dataframe.columns if okay_masking[0,c]])
    return okay


def prepare_data(predictors, targets, rand_seed=731):
    """
    Complete the preparations needed on the data, including
    * fixing implicitly imputed values (see undo_inferred_zero)
    * removing excessively missing data columns (see clear_excess_nullcols)
    * splitting into train and test tests
    * standarding all predictors and responses against the mean and std of the training set
    * imputing any missing (NaN) values to avoid downstream processing issues, again based on training set
    """
    
    predictors = undo_inferred_zero(predictors)
    predictors = clear_excess_nullcols(predictors)
    
    tt_split = model_select.train_test_split(
        predictors,
        targets,
        test_size=0.2,
        random_state=rand_seed
    )
    x_train, x_test, y_trains, y_tests = tt_split
    
    x_standardizer = PolarsStandardizer(x_train)
    y_standardizer = PolarsStandardizer(y_trains)
    
    stand_x_train = x_standardizer.standardize_numeric(x_train)
    stand_y_trains = y_standardizer.standardize_numeric(y_trains)
    
    
    stand_x_train_numeric = stand_x_train.to_dummies(["Sex", "Strain"])
    
    imputer = KNNImputer(keep_empty_features=False)
    imputed_x_train = imputer.fit_transform(stand_x_train_numeric)
    
    stand_x_test = x_standardizer.standardize_numeric(x_test)
    stand_y_tests = y_standardizer.standardize_numeric(y_tests)
    
    stand_x_test_numeric = stand_x_test.to_dummies(["Sex", "Strain"])
    
    imputed_x_test = imputer.transform(stand_x_test_numeric)
    
    x_colnames = list(imputer.get_feature_names_out(stand_x_train_numeric.columns))
    
    prepped_x_train = pl.from_numpy(imputed_x_train, schema=x_colnames)
    prepped_x_test = pl.from_numpy(imputed_x_test, schema=x_colnames)
    
    return prepped_x_train, prepped_x_test, stand_y_trains, stand_y_tests


def build_horseshoe_model(train_x, train_y):
    """
    Constructs a Bayesian linear regression with a 
    Horseshoe prior that encourages sparsity in coefficients.
    For reasons of sampling efficiency, this prior has been 
    constructed using a non-centered parameterization.
    Previously based on the paramerization described at
    https://mellorjc.github.io/HorseshoePriorswithpymc3.html
    and based on 
    https://arxiv.org/abs/1610.05559
    
    Current revision follows either
    https://austinrochford.com/posts/2021-05-29-horseshoe-pymc3.html#fn2
    or
    https://projecteuclid.org/journalArticle/Download?urlId=10.1214%2F17-EJS1337SI
    much more closely.
    """
    
    D_params = train_x.shape[1]
    n_measures = train_x.shape[0]
    
    # hyperparameter - expected number of relevant (non-zero) coefficients in sparse model
    #exp_rel = 300
    #exp_rel = 30 # be more aggressive when ruling out coefficients
    exp_rel = 3000 # a too strong regularization will drive all coefficients to zero
    
    # hyperparameter - controls how aggressive the sparsity promotion is
    #scale_global = 1
    # based on Piironen & Vehtari 2016 recommendation
    scale_global = exp_rel / ((D_params - exp_rel) * np.sqrt(n_measures))
    
    
    with pm.Model() as model:
        x = pm.Data("x", train_x)
        ydat = pm.Data("ydata", train_y)
        
        # prior for noise
        sigma = pm.HalfNormal(
            "sigma",
            sigma=0.5 # standardizing inputs and outputs - expect relatively small noise
        )
        
        #global_shrinkage parameter - tau
        global_shrink = pm.HalfCauchy(
            "global_shrink",
            beta= scale_global * sigma
        )
        
        #component of weights in non-centered parameterization
        beta=pm.Normal("beta",mu=0,tau=1,shape=train_x.shape[1])
        
        #component of local shrinkage parameter in non-centered parameterization - lambda
        local_shrink1 = pm.HalfCauchy(
            "local_shrink1",
            beta=1,
            shape=train_x.shape[1]
        )
        
        # component of local shrinkage in non-centered parameterization
        c_sq = pm.InverseGamma(
            "c_sq",
            alpha=1,
            beta=1
        )
        
        # complete local shrinkage parameter
        local_shrink = pm.Deterministic(
            'local_shrink',
            local_shrink1 * pm.math.sqrt(c_sq / (c_sq + (global_shrink**2 + local_shrink1**2)))
        )
        
        # regression weights
        weights = pm.Deterministic(
            "weights",
            beta * global_shrink * local_shrink
        )
        
        #intercept can be determined mostly by data, no reason to expect zero value
        icpt = pm.Normal('icpt', mu=0, sigma=5)
        
        # need expression, though not necessarily `y` assignee
        y = pm.Normal(
            'y',
            mu=pm.math.dot(x, weights) + icpt,
            sigma=sigma,
            observed=ydat
        )
    return model

if __name__ == '__main__':
    table = load_alldata()
    targets = collect_targets(table)
    predictors = collect_predictors(table)
    x_train, x_test, y_trains, y_tests = prepare_data(predictors, targets)
    train_sk_p16, train_sk_p16_x = clear_null_resp(y_trains["SK p21"], x_train)
    test_sk_p16, test_sk_p16_x = clear_null_resp(y_tests["SK p21"], x_test)
    model = build_horseshoe_model(train_sk_p16_x, train_sk_p16)
    compiled_model = nutpie.compile_pymc_model(model, backend="jax", gradient_backend="jax")
    trace = nutpie.sample(compiled_model, tune=2_000, draws=6_000, low_rank_modified_mass_matrix=True)
    
    ##  use variational inference because sampling is far too slow with models this big
    # with model:
        # prior = pm.sample_prior_predictive()
        # approx = pm.fit(100_000)
    
    # vi_trace = approx.sample(2_000)
    # vi_trace.extend(prior)
    
    # with model:
        # postpred = pm.sample_posterior_predictive(vi_trace)
        # vi_trace.extend(postpred)
        # loglike = pm.compute_log_likelihood(vi_trace)
    
    # with model:
        # pm.set_data({'x': test_sk_p16_x, 'ydata': test_sk_p16})
        # oos_preds = pm.sample_posterior_predictive(vi_trace, predictions=True)
    
    # vi_trace.extend(oos_preds)
    
    
    _y_true = vi_trace.predictions_constant_data["ydata"].expand_dims({"placehold":1}).values
    _y_pred = vi_trace.predictions.stack(sample=("chain","draw"))["y"].values.T
    r2_obj = az.r2_score(_y_true, _y_pred)
    print("R2 score", round(r2_obj['r2'], 5), "R2 stddev", round(r2_obj['r2_std'], 5))
    