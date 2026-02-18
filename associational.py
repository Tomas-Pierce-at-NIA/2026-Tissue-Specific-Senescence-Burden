
import seaborn as sb
import nutpie
import pymc as pm
import arviz as az
from matplotlib import pyplot
import polars as pl
import numpy as np
from polars import selectors as cs
from sklearn import model_selection as model_select
from sklearn.impute import KNNImputer
from sklearn import linear_model
from sklearn import decomposition as decomp

SERUM = "data/Mouse Serum Samples (280 samples)_Protein_Group_Panel.tsv"
MULTIORGAN = "data/Multiorgan_senescence_and_Olink.csv"
SAMPLE_DESC = "data/Sample description for Seer.xlsx"

# maximum fraction of nullness we are willing to tolerate
# before we will not even attempt interpolation / imputation
NULL_TOL = 0.1

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
    # we know with confidence that the only nulls in this table
    # are process controls (helpful experimentally, but irrelevant here)
    clean = table.drop_nulls()
    
    # consistency more than efficiency here
    return clean.lazy()
    

# the files are combined this way, not my fault
def read_multiorgan_olink():
    table = pl.scan_csv(
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
         },
         null_values=['FALSE', 'excluded', 'NA', 'no image', 'needs second IHC'],
         missing_utf8_is_empty_string = False
    )
    return table

# def read_multiorgan_olink():
    # table = pl.read_excel(
         # MULTIORGAN,
         # engine='openpyxl',
         # schema_overrides = {
             # "Animal Tag": pl.String,
             # "SK p16": pl.Float64,
             # "SK p21": pl.Float64,
             # "SK gH2AX": pl.Float64,
             # "LIV p16": pl.Float64,
             # #"LIV p21": pl.Float64,
             # "LIV gH2AX": pl.Float64,
             # "SCO p16": pl.Float64,
             # "SCO p21": pl.Float64,
             # "SCO gH2AX": pl.Float64,
             # "OV p16": pl.Float64,
             # "OV p21": pl.Float64,
             # "OV gH2AX": pl.Float64,
             # "Cxcl1": pl.Float64,
         # },
         # read_options = {'whitespace_as_null': True}
    # )
    # return table

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
    # we know that this procedure specifically will implicitly impute zeros (unjustified)
    # so if we fix that here we don't have to sacrifice real zero-able values to ensure we aren'table
    # imputing inappropriately
    dis_imputed = pivot.select(pl.all().replace(old=0.0, new=None))
    return dis_imputed

def combine(samp_desc, multi_olink, serum):
    combo1 = multi_olink.join(
        samp_desc,
        on=["Animal Tag", "Age (weeks)"],
        how="inner",
        validate="1:1"
    )
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

def collect_targets_female(alldata):
    "special casing because whether ovary collection possible depends on animal sex"
    femtarget = (alldata
      .filter(pl.col('Sex').eq('F'))
      .select(
        cs.ends_with("p16"),
        cs.ends_with("p21"),
        cs.ends_with("gH2AX")
      )
    )
    return femtarget.collect()


def collect_predictors_female(alldata):
    "only female vars relevant for female-only data"
    femdata = (
        alldata
        .filter(pl.col('Sex').eq('F'))
        .select(
            cs.exclude(
                cs.ends_with("p16"),
                cs.ends_with("p21"),
                cs.ends_with("gH2AX")
            )
        )
    )
    return femdata.collect()  


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
            cs.numeric().std() + pl.when(cs.numeric().std().eq(0.0)).then(1e-11).otherwise(0.0)
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
    
    imputer = KNNImputer().set_output(transform='polars')
    # want to learn sparse representation of components matrix,
    # and can only have up to #measures linear independent components
    nmf = decomp.NMF(n_components=280, alpha_W=0.01, l1_ratio=0.4).set_output(transform='polars')
    
    x_train_num = x_train.select(cs.numeric())
    x_train_cat = x_train.select(pl.col('Sex'), pl.col('Strain'))
    x_train_imputed = imputer.fit_transform(x_train_num)
    x_train_comps = nmf.fit_transform(x_train_imputed)
    x_train_decomp = pl.concat([x_train_comps, x_train_cat], how="horizontal")
    x_standardizer = PolarsStandardizer(x_train_decomp)
    x_train_std_decomp = x_standardizer.standardize_numeric(x_train_decomp)
    #x_train_std_decomp_num = x_train_std_decomp.to_dummies(['Sex', 'Strain'])
    
    x_test_num = x_test.select(cs.numeric())
    x_test_cat = x_test.select(pl.col('Sex'), pl.col('Strain'))
    x_test_imputed = imputer.transform(x_test_num)
    x_test_comps = nmf.transform(x_test_imputed)
    x_test_decomp = pl.concat([x_test_comps, x_test_cat], how="horizontal")
    x_test_std_decomp = x_standardizer.standardize_numeric(x_test_decomp)
    #x_test_std_decomp_num = x_test_std_decomp.to_dummies(['Sex', 'Strain'])
    
    return x_train_std_decomp, x_test_std_decomp, y_trains, y_tests



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
    
    # wizard shit
    # control how spread out non-zero coefficients are expected to be
    scale = 2
    deg_free = 4
    
    # expected number of relevant variables
    exp_rel = 3000
    
    
    with pm.Model() as model:
        x = pm.Data("x", train_x)
        ydat = pm.Data("ydata", train_y)
        
        sigma = pm.HalfNormal('sigma', sigma=2.5)
        
        #global shrinkage
        tau_0 = (exp_rel * sigma) / ((D_params - exp_rel) * np.sqrt(n_measures))
        gbl_shrink = pm.HalfCauchy('global_shrink', beta=tau_0)
        
        #local shrinkage
        c2 = pm.InverseGamma('c2', alpha=deg_free/2, beta=deg_free*(scale**2)/2)
        lcl_shrink = pm.HalfCauchy('lcl_shrink', beta=1, shape=train_x.shape[1])
        local_shrink = lcl_shrink * pm.math.sqrt(c2 / (c2 + gbl_shrink**2 * lcl_shrink**2))
        
        #coefficient effect component
        beta = pm.Normal('beta', mu=0, sigma=1, shape=train_x.shape[1])
        
        #fully constructed weights
        weights = beta*local_shrink*gbl_shrink
                
        # need expression, though not necessarily `y` assignee
        y = pm.Normal(
            'y',
            mu=pm.math.dot(x, weights),
            sigma=sigma,
            observed=ydat
        )
    return model

if __name__ == '__main__':
    table = load_alldata()
    targets = collect_targets(table).select(cs.exclude(cs.starts_with("OV")))
    predictors = collect_predictors(table)
    
    femtargets = collect_targets_female(table)
    fempredictors = collect_predictors_female(table)
    
    rng_seeds = np.random.randint(0, 1000, 10):
        x_train, x_test, y_trains, y_tests = prepare_data(predictors, targets)
    
    
    assert False
    
    # display missingness
    fig, ax = pyplot.subplots(figsize=(12,6))
    sb.barplot(x=targets.columns, y=targets.null_count().to_numpy()[0,:], ax=ax)
    pyplot.title("Counts of non-observed tissue senescence markers (280 samples total)")
    pyplot.show()
    
    fig, ax = pyplot.subplots(figsize=(12,6))
    sb.barplot(x=femtargets.columns, y=femtargets.null_count().to_numpy()[0,:], ax=ax)
    pyplot.title("Counts of non-observed tissue senescence markers for females (144 samples total)")
    pyplot.show()
    
    # show distributions
    fig, ax = pyplot.subplots(figsize=(12,6))
    side_targ = targets.unpivot()
    sb.stripplot(side_targ.drop_nulls(), x="variable", y="value", ax=ax, size=2)
    sb.boxplot(side_targ.drop_nulls(), x="variable", y="value", ax=ax)
    pyplot.title("Distribution of observed tissue senescence markers")
    pyplot.show()
    
    fig, ax = pyplot.subplots(figsize=(12,6))
    k = side_targ.select(pl.col('variable'), pl.col('value').log1p())
    sb.stripplot(k.drop_nulls(), x="variable", y="value", ax=ax, size=2)
    sb.boxplot(k.drop_nulls(), x="variable", y="value", ax=ax)
    pyplot.title("Distribution of log1p [ie ln(x + 1)] of tissue senescence markers")
    pyplot.show()
    
    fig, ax = pyplot.subplots(figsize=(12,6))
    side_fem = femtargets.unpivot()
    sb.stripplot(side_fem.drop_nulls(), x="variable", y="value", ax=ax, size=2)
    sb.boxplot(side_fem.drop_nulls(), x="variable", y="value", ax=ax)
    pyplot.title("Distribution of observed tissue senescence markers for females")
    pyplot.show()
    
    fig, ax = pyplot.subplots(figsize=(12,6))
    k2 = side_fem.select(pl.col('variable'), pl.col('value').log1p())
    sb.stripplot(k2.drop_nulls(), x="variable", y="value", ax=ax, size=2)
    sb.boxplot(k2.drop_nulls(), x="variable", y="value", ax=ax)
    pyplot.title("Distribution of log1p [ie ln(x + 1)] of tissue senescence markers for females")
    pyplot.show()
    #targets = targets.select(pl.all().log10())
    
    #predictors = predictors.select(pl.all().log10())
    #assert False
    # x_train, x_test, y_trains, y_tests = prepare_data(predictors, targets)
    
    # for pred_name in targets.columns:
        # train_y2, train_x2 = clear_null_resp(y_trains[pred_name], x_train)
        # test_y2, test_x2 = clear_null_resp(y_tests[pred_name], x_test)
        # enet = linear_model.ElasticNetCV(l1_ratio=[0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99], alphas=30, max_iter=50_000, fit_intercept=True)     
        # enet.fit(train_x2, train_y2)
        # r2 = enet.score(test_x2, test_y2)
        # print(r2, pred_name)
        
    # #train_sk_p16, train_sk_p16_x = clear_null_resp(y_trains["SK p21"], x_train)
    # #test_sk_p16, test_sk_p16_x = clear_null_resp(y_tests["SK p21"], x_test)
    # assert False
    # model = build_horseshoe_model(train_sk_p16_x, train_sk_p16)
    # compiled_model = nutpie.compile_pymc_model(model, backend="jax", gradient_backend="pytensor")
    # trace = nutpie.sample(compiled_model, tune=2_000, draws=6_000, target_accept=0.9)
    
    
    