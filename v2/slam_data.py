
import polars as pl
from polars import selectors as cs

SLAM_EUTH = "data/slam900/Slam_900_phenotypes_combined w fast-fed.xlsx"
SLAM_LONG_SERUM = "data/slam900/slam_900_counts_phenotypes.csv"


def normalize_sample_id(table):
    """normalize the sample ID code to contain no spaces so all are the same format"""
    slamic_index = table.columns.index('SLAMICS ID')
    table.replace_column(slamic_index, table['SLAMICS ID'].str.replace(' ', ''))


def read_euth() -> pl.DataFrame:
    """Reads in file which records which mice were euthanized and when from slam 900 dataset"""
    table = pl.read_excel(
        SLAM_EUTH,
        schema_overrides={'Euthanasia Note': pl.String}
    )
    table = table.with_columns(
        is_euthanized = pl.col('Euthanasia Note').is_not_null()
    )
    
    # "gets euthanized" table
    ge_tab = (table
              .select(pl.col('ID'),pl.col('is_euthanized'))
              .group_by(['ID'])
              .sum()
              .select(pl.col('ID'),pl.col('is_euthanized').alias('gets_euthanized'))
    )
    
    table = table.join(ge_tab, how="inner", on="ID")
    
    table = table.select(pl.exclude(["Visit", "Euthanasia Note"]))
    
    normalize_sample_id(table)
    
    return table


def read_serum():
    """Reads in the longitudinal serum data taken from the slam 900 dataset"""
    table = pl.read_csv(
        SLAM_LONG_SERUM,
        null_values = ["NA"],
    )
    normalize_sample_id(table)
    return table


def combine_serum_euth(serum_tab, euth_tab):
    """Combine euthansia information with longitudinal serum data"""
    # the way the serum identifies mice is not the way the euthanasia identifies mice
    no_id_euth = euth_tab.select(pl.exclude('ID'))
    return serum_tab.join(no_id_euth, how='left', on=['SLAMICS ID', 'Age', 'Sex', 'Strain'])


def load_slam900_longitudinal():
    """Load in the longitudinal SLAM900 dataset 
    and do initial data setup steps including
    adding the euthanasia information and ensuring
    appropriate data types"""
    euth = read_euth()
    serum = read_serum()
    combo = combine_serum_euth(serum, euth)
    tab = combo.select(pl.exclude(['Visit', 'IDVisit']))
    tab = tab.select(
        pl.col(['SLAMICS ID', 
                'Sex', 
                'Strain', 
                'Age', 
                'ID', 
                'Cohort', 
                'is_euthanized', 
                'gets_euthanized'
                ]),
        pl.exclude(['SLAMICS ID', 
                    'Sex', 
                    'Strain', 
                    'Age', 
                    'ID', 
                    'Cohort', 
                    'is_euthanized', 
                    'gets_euthanized'
                    ]).cast(pl.Float64)
    )
    return tab

