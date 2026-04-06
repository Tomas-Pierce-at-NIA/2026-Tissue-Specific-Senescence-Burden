
import polars as pl

import data
from dataloader import DataLoader
from data_prep import DataPrep
from reduce_collinear_feature import ClusterRepSel
import horseshoe
import slam_data


if __name__ == '__main__':
    dl = DataLoader()
    dprep = DataPrep()
    slam = slam_data.load_slam900_longitudinal()
    pgmap = data.create_protein_gene_mapping()
    long_pgmap = pgmap.filter(pl.col('Gene Names').is_in(pl.lit(slam.columns)))
    
    train_x = dl.get_train_predictors()
    train_x2 = train_x.select(
        [pl.col(c) for c in long_pgmap['Protein Names'].unique() if c in train_x.columns]
    )
    train_x3 = dprep.fit_transform(train_x2)
    
    
    
    