
import associational as assoc
import pymc as pm
import arviz as az
import polars as pl
import numpy as np
import itertools
from collections import deque

# this file implements a simplified version of the Peter-Clarkson algorithm for causal discovery.
# algorithm makes assumption of linear relationships.

class Skeleton:
    "represents the undirected skeleton of a graph between n variables"
    
    def __init__(self, n_vars, start_fully_connected=True):
        """n_vars is number of variables in graph, 
        will start with fully connected skeleton if start_fully_connected,
        will start fully disconnected otherwise
        """
        self.__n_vars = n_vars
        if start_fully_connected:
            self.__grid = np.ones((n_vars, n_vars), dtype=bool)
        else:
            self.__grid = np.zeros((n_vars, n_vars), dtype=bool)
    
    def prune(self, var_i, var_j):
        "prune connection between variable at index i and variable at index j"
        self.__grid[var_i,var_j] = False
        self.__grid[var_j,var_i] = False # needed b/c skeleton, so undirected, so prune both sides
    
    def connect(self, var_i, var_j):
        "connect variables var_i and var_j"
        self.__grid[var_i,var_j] = True
        self.__grid[var_j,var_i] = True
    
    def are_disjointed(self, variables :list[int]) -> bool:
        """Detect whether the variables in the list of variable indices 
           are part of a connected sub-graph
        """
        variables = np.array(variables, dtype=int)
        if len(variables) < 2:
            raise RuntimeError("doesn't make sense to look for connectivity of less than 2 variables")
        
        # breath first search - idea is that for a connected sub-graph,
        # we will be able to reach all var nodes from any of the var nodes,
        # but for a disconnected set, we will not be able to reach at least
        # one of the var nodes from each individual other var node
        was_hit = np.zeros((self.__n_vars,), dtype=bool)
        v0 = variables[0]
        queue = deque()
        queue.append(v0)
        while len(queue) > 0:
            # speeds up the case where most variables are connected but 
            # only care about specific subset being connected,
            # without compromising correctness
            if np.all(was_hit[variables]):
                return False
            item = queue.popleft()
            was_hit[item] = True
            conn_mask = self.__grid[item]
            connected = np.argwhere(conn_mask & (~was_hit))[:,0]
            queue.extend(connected)
        
        return not np.all(was_hit[variables])

def build_model(data):
    """
    construct a simple linear model on input data
    """
    with pm.Model() as model:
        obs = pm.Normal('obs',mu=0,sigma=1,observed = data[:,1:],shape=data[:,1:].shape)
        weights = pm.Normal('weights',mu=0,sigma=1,shape=data.shape[1]-1)
        icpt = pm.Normal('icpt',mu=0,sigma=5)
        sigma = pm.Exponential('sigma', lam=5)
        resp = pm.Normal('resp',mu = icpt + pm.math.matmul(obs, weights),sigma=sigma,observed=data[:,0])
    return model


def prune_phase(skel, my_table):
    var_indices = np.arange(0, my_table.shape[1])
    for subset_size in range(2, my_table.shape[1]):
        subsets = itertools.combinations(var_indices, subset_size)
        for subset in subsets:
            if skel.are_disjointed(subset):
                continue
            sub_table = my_table[:, [int(idx) for idx in subset]]
            local_model = build_model(sub_table)
            with local_model:
                prior = pm.sample_prior_predictive()
                approx = pm.fit(10_000)
                trace = approx.sample(2_000)
                postp = pm.sample_posterior_predictive(trace)
            trace.extend(prior)
            trace.extend(postp)
            for in_localmodel_idx in range(sub_table.shape[1]-1):
                w_post = trace.posterior.sel({'weights_dim_0':in_localmodel_idx})
                bayes_factor = az.bayes_factor(w_post, "weights", ref_val=0)
                # care about the strength of evidence in favor of independence (=0, ie prior) conditional
                # on the other regressors, want to only rule out causal link if evidence
                # of that conditional independence is strong
                if bayes_factor['BF01'] >= 10: # strong evidence of conditional independence
                    target_var = subset[0]
                    indep_var = subset[in_localmodel_idx+1]
                    skel.prune(target_var, indep_var)
    return


if __name__ == '__main__':
    table = assoc.load_alldata()
    targets = assoc.collect_targets(table)
    othervars = assoc.collect_predictors(table)
    othervars = assoc.undo_inferred_zero(othervars)
    othervars = assoc.clear_excess_nullcols(othervars, 0.05)
    eager_table = pl.concat([targets, othervars], how='horizontal')
    eager_table = eager_table.to_dummies(['Sex', 'Strain'], drop_first=True)
    eager_table = eager_table.select((pl.all() - pl.all().mean())/pl.all().std())
    skel = Skeleton(eager_table.shape[1])
    prune_phase(skel, eager_table)
    