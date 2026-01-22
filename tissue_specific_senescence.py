# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 12:22:40 2026

@author: piercetf
"""

import pymc as pm
from sklearn import model_selection as model_sel
from sklearn import linear_model as lin
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

if __name__ == '__main__':
    organs, olink, serum_proteome, sample_desc = load_data()
    table1 = ready_table1(organs, serum_proteome, sample_desc)
    table1_present = filter_excess_missing(table1)
    table1p_numeric = (table1_present
                       # don't need to track b/c all samples from diff animals - iid
                       .select(cs.exclude(["Animal Tag", "Sample Name"]))
                       # need to enable learners which don't tolerate strings
                       .to_dummies(["Strain", "Sex"])
                       )
    
    '''
    y_variables = table1p_numeric.select(cs.ends_with("p16"),
                                         cs.ends_with("p21"),
                                         cs.ends_with("gH2AX"))
    x_variables = table1p_numeric.select(cs.exclude([cs.ends_with("p16"),
                                         cs.ends_with("p21"),
                                         cs.ends_with("gH2AX")]))
    '''
    
    
    