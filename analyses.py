
import data
import less_collinear_features
import numpy as np
from sklearn import impute, linear_model, preprocessing
import polars as pl
from polars import selectors as cs
from matplotlib import pyplot
import seaborn as sb

def numerize(df):
    return df.select(cs.numeric(), pl.col('Sex').eq('F').alias('is_F'), pl.col('Strain').eq('B6').alias('is_B6'))

def standardize_noncat(df, std, fit=False):
    noncat = df.select(cs.exclude(['is_F', 'is_B6']))
    cat = df.select(pl.col('is_F'), pl.col('is_B6'))
    if fit:
        noncat_std = std.fit_transform(noncat)
    else:
        noncat_std = std.transform(noncat)
    rejoined = pl.concat([noncat_std, cat], how='horizontal')
    return rejoined

def clust_rep_nonspecial(df, clust_rep, fit=False):
    if 'Age (weeks)' in df.columns:
        nonspecial = df.select(cs.exclude(['Age (weeks)', 'is_F', 'is_B6']))
        special = df.select(pl.col('Age (weeks)'), pl.col('is_F'), pl.col('is_B6'))
    else:
        nonspecial = df.select(cs.exclude(['is_F', 'is_B6']))
        special = df.select(pl.col('is_F'), pl.col('is_B6'))
    if fit:
        clust_repped = clust_rep.fit_transform(nonspecial)
    else:
        clust_repped = clust_rep.transform(nonspecial)
    rejoined = pl.concat([clust_repped, special], how='horizontal')
    return rejoined

def evaluate_enets():
    
    table = {'rng_seed': [],
             'female_only': [],
             'target': [],
             'transform': [],
             'score': [],
             'includes_age': []
    }
    
    rng = np.random.default_rng(2026_18_02)
    rng_seeds = rng.integers(0, 1000, 5)
    fem_only = [False, True]
    include_age = [True, False]
    for seed in rng_seeds:
        for rule in fem_only:
            for inclusion in include_age:
                split = data.prepare_datasets(fem_only=rule, rng_seed=seed)
                x_train, x_test, y_trains, y_tests = split
                if not inclusion:
                    x_train = x_train.select(pl.exclude("Age (weeks)"))
                    x_test = x_test.select(pl.exclude("Age (weeks)"))
                for cname in y_trains.columns:
                    if "OV" in cname and not rule:
                        # no point trying to fit ovary models unless only using female data
                        continue
                    print("{} {} {}".format(seed, rule, cname))
                    ytrain2, xtrain2 = data.clear_null_response(y_trains[cname], x_train)
                    ytest2, xtest2 = data.clear_null_response(y_tests[cname], x_test)
                    
                    # allow fitting log-linear models when the minimum value is zero
                    epsilon = 1e-14
                    log_ytrain = (ytrain2 + epsilon).log10()
                    log_ytest = (ytest2 + epsilon).log10()
                    
                    xtrain3 = numerize(xtrain2)
                    xtest3 = numerize(xtest2)
                    
                    imputer = impute.SimpleImputer(strategy='median').set_output(transform='polars')
                    
                    xtrain4 = imputer.fit_transform(xtrain3)
                    xtest4 = imputer.transform(xtest3)
                    
                    #clust_rep = less_collinear_features.SelectClusterRep(512)
                    #xtrain4 = clust_rep_nonspecial(xtrain4, clust_rep, fit=True)
                    #xtest4 = clust_rep_nonspecial(xtest4, clust_rep, fit=False)
                    
                    std_izer = preprocessing.StandardScaler().set_output(transform='polars')
                    xtrain5 = standardize_noncat(xtrain4, std_izer, fit=True)
                    xtest5 = standardize_noncat(xtest4, std_izer, fit=False)
                    
                    lin_model = linear_model.ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], 
                                                          alphas=100,
                                                          max_iter=5_000,
                                                          tol=1e-3,
                                                          selection='random',
                                                          random_state=seed,
                                                          n_jobs=-1
                                                          )
                    loglin_model = linear_model.ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], 
                                                             alphas=100,
                                                             max_iter=5_000,
                                                             tol=1e-3,
                                                             selection='random',
                                                             random_state=seed,
                                                             n_jobs=-1
                                                             )
                    
                    lin_model.fit(xtrain5, ytrain2)
                    loglin_model.fit(xtrain5, log_ytrain)
                    
                    lin_score = lin_model.score(xtest5, ytest2)
                    loglin_score = loglin_model.score(xtest5, log_ytest)
                    
                    table['rng_seed'].append(seed)
                    table['female_only'].append(rule)
                    table['target'].append(cname)
                    table['transform'].append('ident')
                    table['score'].append(lin_score)
                    table['includes_age'].append(inclusion)
                    
                    
                    table['rng_seed'].append(seed)
                    table['female_only'].append(rule)
                    table['target'].append(cname)
                    table['transform'].append('log')
                    table['score'].append(loglin_score)                    
                    table['includes_age'].append(inclusion)
                    #assert False
    
    table_df = pl.from_dict(table)
    return table_df

if __name__ == '__main__':
    table = evaluate_enets()
    
    sb.boxplot(table.filter(pl.col('transform').eq('log') & pl.col('female_only').eq(False)),
               x='target',
               y='score',
               hue='includes_age'
              )
    pyplot.suptitle("variance from randomizing train-test split")
    pyplot.title("elastic net log-linear model R2 scores out-of-sample (both sexes)")
    pyplot.show()
    
    sb.boxplot(table.filter(pl.col('transform').eq('log') & pl.col('female_only').eq(True)),
               x='target',
               y='score',
               hue='includes_age'
              )
    pyplot.suptitle("variance from randomizing train-test split")
    pyplot.title("elastic net log-linear model R2 scores out-of-sample (females only)")
    pyplot.show()
    
    sb.boxplot(table.filter(pl.col('transform').eq('ident') & pl.col('female_only').eq(False)),
               x='target',
               y='score',
               hue='includes_age'
              )
    pyplot.suptitle("variance from randomizing train-test split")
    pyplot.title("elastic net linear model R2 scores out of sample (both sexes)")
    pyplot.show()
    
    sb.boxplot(table.filter(pl.col('transform').eq('ident') & pl.col('female_only').eq(True)),
               x='target',
               y='score',
               hue='includes_age'
              )
    pyplot.suptitle("variance from randomizing train-test split")
    pyplot.title("elastic net linear model R2 scores out of sample (females only)")
    pyplot.show()
    