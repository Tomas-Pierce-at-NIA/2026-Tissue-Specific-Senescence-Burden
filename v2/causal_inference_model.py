
from matplotlib import pyplot
import numpy as np
import polars as pl

from reduce_collinear_feature import ClusterRepSel
from dataloader import DataLoader
from data_prep import DataPrep
import data

from causallearn.search.ConstraintBased.PC import pc

dl = DataLoader()
train_x = dl.get_train_predictors()
train_y = dl.get_train_target('SK gH2AX')

dp = DataPrep()

train_x2 = dp.fit_transform(train_x)

crs = ClusterRepSel(1024)

train_x3 = crs.fit_transform(train_x2)

train_data = train_x3.with_columns(train_y)

causal_graph = pc(train_data.to_numpy(), mvpc=True, indep_test="mv_fisherz", show_progress=True)

