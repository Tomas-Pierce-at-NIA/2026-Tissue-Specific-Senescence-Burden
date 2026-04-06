
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
    
    train_y = dl.get_train_target('SK gH2AX')
    
    train_y2, train_x4 = data.clear_null_response(train_y, train_x3)
    
    train_demo = dl.get_train_demographics()
    _, train_demo2 = data.clear_null_response(train_y, train_demo)
    train_demo3 = horseshoe.demo_transform(dl.get_demographic_dimensions(), train_demo2)
    
    clust_sel = ClusterRepSel(1024)
    train_x5 = clust_sel.fit_transform(train_x4)
    
    if 'Age (weeks)' not in train_x5.columns:
        train_x6 = pl.concat([train_x5, train_x4.select(pl.col('Age (weeks)'))], how='horizontal')
    else:
        train_x6 = train_x5
    
    model = horseshoe.horseshoe_model(train_x6, train_demo3, train_y2, 262)
    
    
    
    