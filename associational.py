
#import pandas as pd
import polars as pl
import numpy as np
from polars import selectors as cs
from sklearn import model_selection as model_select

SERUM = "data/Mouse Serum Samples (280 samples)_Protein_Group_Panel.tsv"
MULTIORGAN = "data/Multiorgan_senescence_and_Olink.xlsx"
SAMPLE_DESC = "data/Sample description for Seer.xlsx"

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
        values=pl.col("Normalized Intensity (Log10)"),
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
    ).select(cs.numeric, 
             pl.col("Sex"), 
             pl.col("Strain")
    ).collect()

