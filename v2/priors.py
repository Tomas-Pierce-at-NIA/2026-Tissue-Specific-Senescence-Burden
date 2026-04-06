# this file helps us derive priors from external datasets
# which let us inform our priors when analyzing this dataset

import polars as pl
from polars import selectors as cs

# These files are supplemental tables from the senesence catalog paper.
# link is https://www.biorxiv.org/content/10.64898/2026.02.05.703986v1
# We will use these to justify a prior parameter for the number of serum proteins
# associated with tissue senesence burden.
# contains lists of proteins positively associated with senesence on a per cell-type basis
SENCAT_SUP5 = "data/prior_informing/SenCAT_SuppTable5.csv"
# contains lists of proteins negatively associated with senesence on a per cell-type basis
SENCAT_SUP6 = "data/prior_informing/SenCAT_SuppTable6.csv"

def load_sencat_table(fpath):
    table = pl.scan_csv(fpath, skip_rows=1)
    return table.select(cs.exclude(pl.col('Cell')))

def count_detected_associations(tab):
    return tab.select(pl.all().is_not_null().sum())

def select_cell_types(tab):
    return tab.select(cs.exclude(cs.contains(' ')))

def get_assoc_counts(fpath):
    tab = load_sencat_table(fpath)
    tab = select_cell_types(tab)
    tab = count_detected_associations(tab)
    return tab.collect()

def avg_assoc_count(tab):
    mean = tab.to_numpy().mean()
    return round(mean)

if __name__ == '__main__':
    plus_assocs = get_assoc_counts(SENCAT_SUP5)
    plus_count = avg_assoc_count(plus_assocs)
    minus_assocs = get_assoc_counts(SENCAT_SUP6)
    minus_count = avg_assoc_count(minus_assocs)
    total_assocs = plus_assocs + minus_assocs
    total_count = avg_assoc_count(total_assocs)
    
    print("""Marginal number of positively senesence associated proteins, 
that is, number of positively senescence associated proteins
averaged over the different cell types used in in Anterillas et al 2026 (uniform weighting):
{}""".format(plus_count)
    )
    
    print("""Marginal number of negatively senesence associated proteins,
that is, number of negatively senescence associated proteins
averaged over the different cell types used in Anterillas et al 2026 (uniform weighting):
{}""".format(minus_count)
    )
    
    print("""Marginal number of senesence associated protein,
that is, number of senescence associated proteins 
averaged over the different cell types used in Anterillas et al 2026 (uniform weighting):
{}""".format(total_count)
    )
    
    print("""Average of marginal numbers of positively and negatively senesence associated 
proteins as previously defined:
{}""".format((plus_count + minus_count) // 2)
    )
    
    