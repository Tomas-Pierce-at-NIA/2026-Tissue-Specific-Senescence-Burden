
import json

import polars as pl

import pymc as pm
import nutpie
import arviz as az
from matplotlib import pyplot
import numpy as np


from reduce_collinear_feature import ClusterRepSel
from dataloader import DataLoader
from data_prep import DataPrep
import data

def demo_transform(dims, demo_df):
    return demo_df.select(pl.col('Sex').eq(pl.lit(dims['Sex'][1])),
                          pl.col('Strain').eq(pl.lit(dims['Strain'][1]))
                          ).cast(pl.Int64).to_numpy() # work around known bug in pymc (issue #6874)

def horseshoe_model(train_x, train_demo, train_y, exp_rel, nz_df=3, nz_scale=5):

    dims = DataLoader.get_demographic_dimensions()
    
    train_n, d_params = train_x.shape
    
    global_df = 3
    local_df = 3
    
    with pm.Model(coords=dims) as model:
        
        x = pm.Data('x', train_x)
        ydata = pm.Data('ydata', train_y)
        
        # per-subpopulation intercept term
        
        demo = pm.Data('demo', train_demo)
        #breakpoint()
        icpt = pm.Normal('icpt', mu=0, sigma=5, dims=['Sex', 'Strain'])
        
        sigma = pm.HalfNormal('sigma', sigma=2.5)
        
        tau0 = (exp_rel * sigma) / ((d_params - exp_rel) * np.sqrt(train_n))
        
        global_shrink = pm.HalfStudentT('global_shrink', nu=global_df, sigma=tau0)
        
        lcl_shrink = pm.HalfStudentT('lcl_shrink', nu=local_df, sigma=1, shape=d_params)
        c2 = pm.InverseGamma('c2', alpha=nz_df/2, beta=nz_df*(nz_scale**2)/2)
        local_shrink = lcl_shrink*pm.math.sqrt(c2 / (c2 + (global_shrink**2)*(lcl_shrink**2)))
        
        beta = pm.Normal('beta', mu=0, sigma=1, shape=d_params)
        
        weights = pm.Deterministic('weights', beta*local_shrink*global_shrink)
        
        lin = icpt[demo[:,0], demo[:,1]] + pm.math.dot(x, weights)
        
        y = pm.Normal('y', mu=lin, sigma=sigma, observed=ydata)
    
    return model


def identity(target_data):
    return target_data


def log10_transform(target_data):
    return target_data.log10()

def sqrt_transform(target_data):
    return target_data.sqrt()
#def offset_log10(target_data, offset=1e-15):
    #return (target_data + offset).log10()

def main(target_name, resp_transform=identity, female_only=False):
    dl = DataLoader(female_only=female_only)
    dprep = DataPrep()
    train_x = dl.get_train_predictors()
    train_demo = dl.get_train_demographics()
    train_y = dl.get_train_target(target_name)
    
    train_x2 = dprep.fit_transform(train_x)
    
    train_y2, train_x3 = data.clear_null_response(train_y, train_x2)
    _, train_demo2 = data.clear_null_response(train_y, train_demo)
    train_demo3 = demo_transform(dl.get_demographic_dimensions(), train_demo2)
    
    clust_sel = ClusterRepSel(1024)
    train_x4 = clust_sel.fit_transform(train_x3)
    
    if 'Age (weeks)' not in train_x4.columns:
        train_x5 = pl.concat([train_x4, train_x3.select(pl.col('Age (weeks)'))], how='horizontal')
    else:
        train_x5 = train_x4
    
    train_y3 = resp_transform(train_y2)
    
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC3273898/
    hs = horseshoe_model(train_x5, train_demo3, train_y3, 262)
    
    compiled_model = nutpie.compile_pymc_model(hs, backend='jax', gradient_backend='jax')
    trace = nutpie.sample(compiled_model, target_accept=0.95, tune=1000, draws=2000)
    
    with hs:
        prior = pm.sample_prior_predictive()
        ppc = pm.sample_posterior_predictive(trace)
        trace.extend(prior)
        trace.extend(ppc)
        loglike = pm.compute_log_likelihood(trace)
        trace.extend(loglike)
    
    test_y = dl.get_test_target(target_name)
    test_x = dl.get_test_predictors()
    test_demo = dl.get_test_demographics()
    
    test_x2 = dprep.transform(test_x)
    test_y2, test_x3 = data.clear_null_response(test_y, test_x2)
    _, test_demo2 = data.clear_null_response(test_y, test_demo)
    
    test_demo3 = demo_transform(dl.get_demographic_dimensions(), test_demo2)
    
    test_x4 = clust_sel.transform(test_x3)
    
    if 'Age (weeks)' not in test_x4.columns:
        test_x5 = pl.concat([test_x4, test_x3.select(pl.col('Age (weeks)'))], how='horizontal')
    else:
        test_x5 = test_x4
    
    test_y3 = resp_transform(test_y2)
    
    with hs:
        pm.set_data({'x': test_x5, 'demo': test_demo3, 'ydata': test_y3})
        preds = pm.sample_posterior_predictive(trace, predictions=True)
    
    trace.extend(preds)
    
    with open("out/features_used.json", "w") as feat_hand:
        json.dump(train_x5.columns, feat_hand)
    
    trace.to_netcdf("out/trace.netcdf")
    
    return trace


if __name__ == '__main__':
    target = 'SK gH2AX'
    tr = main(target, identity)
    #target = 'SK p21'
    #tr = main(target, log10_transform)
    #target = 'SK p16'
    #tr = main(target, log10_transform)
    #target = 'OV gH2AX'
    #tr = main(target, identity, True)
    #tr = main(target, identity)
    
    