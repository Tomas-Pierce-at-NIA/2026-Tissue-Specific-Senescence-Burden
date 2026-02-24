
import polars as pl
from polars import selectors as cs
import numpy as np
from sklearn.model_selection import ShuffleSplit

SERUM = "data/Mouse Serum Samples (280 samples)_Protein_Group_Panel.tsv"
MULTIORGAN = "data/Multiorgan_senescence_and_Olink.csv"
SAMPLE_DESC = "data/Sample description for Seer.xlsx"

TOL = 0.1
TEST_FRAC = 0.2


def read_sampledesc():
    """Reads file describing samples, needed to connect serum samples to tissue measurements"""
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
    """Reads file which combines serum OLINK with tissue senescence markers, labeled by animal tag"""
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


def gene_lookup_by_name():
    """use the serum file to produce a lookup table to translate between protein names
    and gene names.
    """
    lazy = pl.scan_csv(SERUM, separator='\t')
    lookup = lazy.select(pl.col('Protein Names'), pl.col('Gene Names'))
    dedup = lookup.unique()
    return dedup


def read_serum():
    """Reads file with serum proteomics measurements, converts format to be 
    column per protein and ensures that in-appropriate zero imputations are removed."""
    lazy = pl.scan_csv(SERUM,
                       separator="\t"
                       )
    prot_list = lazy.select(pl.col("Protein Names").unique()).collect()["Protein Names"]
    #gene_list = lazy.select(pl.col("Gene Names").unique()).collect()["Gene Names"]
    pivot = lazy.pivot(
        on=pl.col("Protein Names"),
        #on=pl.col("Gene Names"),
        on_columns=prot_list,
        #on_columns=gene_list,
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
    """Performs the task of matching 
    (serum MS) -- (sample name) -- (animal ID) -- (tissue senescence markers + OLINK)
    """
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


def get_targets(alldata, female_only=False):
    """Lazily select tissue senescence markers (targets),
    filtering for female samples only at caller option
    """
    if female_only:
        reldata = alldata.filter(pl.col('Sex').eq('F'))
    else:
        reldata = alldata
    targets = reldata.select(cs.ends_with("p16"), cs.ends_with("p21"), cs.ends_with("gH2AX"))
    return targets


def get_predictors(alldata, female_only=False):
    """Lazily select all variables except tissue senescence markers (predictors)
    filtering for female samples only at caller option
    """
    if female_only:
        reldata = alldata.filter(pl.col('Sex').eq('F'))
    else:
        reldata = alldata
    selector = cs.exclude(cs.ends_with("p16"), cs.ends_with("p21"), cs.ends_with("gH2AX"))
    predictors = reldata.select(selector)
    return predictors


def get_subset_indices(df, indices):
    """Helper function to lazily filter choosing rows based on a collection
    of row indices. Useful for lazily conducting train-test split
    """
    row_labeled = df.with_row_index()
    sub_df = row_labeled.filter(pl.col("index").is_in(indices))
    return sub_df


def train_test_split(predictors, targets, test_frac=TEST_FRAC, rng_seed=None):
    """Split lazy dataframes into separate training and testing sets.
    """
    if rng_seed is None:
        from os import urandom
        rand_bytes = urandom(4)
        rng_seed = int.from_bytes(rand_bytes)
    n_obs = predictors.select(pl.len()).collect().item()
    empty = np.zeros((n_obs,1))
    splitter = ShuffleSplit(test_size=test_frac, random_state=rng_seed)
    splits = splitter.split(empty)
    train_idx, test_idx = next(splits)
    
    train_pred = get_subset_indices(predictors, train_idx)
    test_pred = get_subset_indices(predictors, test_idx)
    
    train_targets = get_subset_indices(targets, train_idx)
    test_targets = get_subset_indices(targets, test_idx)
    
    return train_pred, test_pred, train_targets, test_targets


def clear_excess_nullcols(df, tol=TOL):
    """Return new dataframe lazily removing columns which exceed 
    tolerance for missing data proportion
    """
    n_obs = df.select(pl.len()).collect().item()
    null_counts = df.null_count()
    null_frac = null_counts.select(pl.all() / pl.lit(n_obs))
    okay_mask = null_frac.select(pl.all() < pl.lit(tol))
    okay_real = okay_mask.collect()
    okay_indices = np.argwhere(okay_real)[:,1]
    col_exprs = [pl.col(okay_real.columns[idx]) for idx in okay_indices]
    return df.select(col_exprs)


def prepare_datasets(fem_only=False, rng_seed=334):
    """Conduct all steps needed to collect 
    training predictors, testing predictors, training targets, testing targets
    filtering for female only data at caller option.
    Uses polars lazyframes for improved speed & efficiency.
    """
    serum = read_serum()
    multi_olink = read_multiorgan_olink()
    samps = read_sampledesc()
    combo = combine(samps, multi_olink, serum)
    no_dupcols = combo.select(cs.exclude(cs.ends_with("_right")))
    targets = get_targets(no_dupcols, fem_only)
    predictors = get_predictors(no_dupcols, fem_only)
    predictors = clear_excess_nullcols(predictors)
    x_train, x_test, y_trains, y_tests = train_test_split(predictors, targets, rng_seed=rng_seed)
    # don't need an index at this point & would interfere with downstream steps
    x_train = x_train.select(pl.exclude("index"))
    x_test = x_test.select(pl.exclude("index"))
    y_trains = y_trains.select(pl.exclude("index"))
    y_tests = y_tests.select(pl.exclude("index"))
    return x_train.collect(), x_test.collect(), y_trains.collect(), y_tests.collect()


def clear_null_response(target, predictors):
    """
    get rid of rows where the target is null
    because the tools aren't set up to deal with that
    """
    valid_mask = ~target.is_null()
    valid_target = target.filter(valid_mask)
    corresp_predictors = predictors.filter(valid_mask)
    return valid_target, corresp_predictors


def numerize_predictors(predictors):
    selected = predictors.select(cs.numeric(), pl.col('Sex'), pl.col('Strain'))
    dummied = selected.to_dummies(['Sex', 'Strain'])
    no_cat_colinear = dummied.select(pl.exclude(['Sex_M', 'Strain_HET3']))
    return no_cat_colinear

