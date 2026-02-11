
import data
import polars as pl
from polars import selectors as cs
from statsmodels import api as sm
from sklearn import metrics
import numpy as np
import seaborn as sb
from matplotlib import pyplot

def main(use_female_only=False):
    x_train, x_test, y_trains, y_tests = data.prepare_datasets(use_female_only)
    x_train_num = x_train.select(cs.numeric())
    x_test_num = x_test.select(cs.numeric())
    
    if not use_female_only:
        targets = ['SK p16', 'SK p21', 'SK gH2AX']
    else:
        targets = ['SK p16', 'SK p21', 'SK gH2AX', 'OV p16', 'OV p21', 'OV gH2AX']
    
    rows = []
    
    for target in targets:
        y_train = y_trains[target]
        y_test = y_tests[target]
        
        knowable_testcases = y_test.is_not_null()
        y_test2 = y_test.filter(knowable_testcases)
        x_test_num2 = x_test_num.filter(knowable_testcases)
        
        for col_idx in range(x_train_num.shape[1]):
            col_train = x_train_num[:, [col_idx]]
            col_test = x_test_num2[:, [col_idx]]
            
            knowable_testinputs = col_test[:,0].is_not_null()
            y_test3 = y_test2.filter(knowable_testinputs)
            col_test2 = col_test.filter(knowable_testinputs)
            
            col_train_np = sm.add_constant(col_train.to_numpy())
            
            ols = sm.OLS(endog=y_train.to_numpy(), exog=col_train_np, missing='drop', hasconst=True)
            result = ols.fit()
            
            col_test_np = sm.add_constant(col_test2.to_numpy())
            preds = result.predict(col_test_np)
            
            oos_r2 = metrics.r2_score(y_true=y_test3, y_pred=preds)
            oos_mse = metrics.mean_squared_error(y_true=y_test3, y_pred=preds)
            oos_mae = metrics.mean_absolute_error(y_true=y_test3, y_pred=preds)
            
            ci = result.conf_int(0.05)[1]
            
            row = {'target': target,
                   'predictor': col_train.columns[0],
                   'coef': result.params[1],
                   'p-value': result.pvalues[1],
                   '95% CI left': ci[0],
                   '95% CI right': ci[1],
                   #'95% CI': result.conf_int(0.05)[1][0],
                   'OOS R2': oos_r2,
                   'OOS MSE': oos_mse,
                   'OOS MAE': oos_mae,
                   'train R2': result.rsquared,
                   'train R2 adj': result.rsquared_adj
                }
            
            rows.append(row)
    return rows

if __name__ == '__main__':
    rows = main()
    table = pl.from_dicts(rows)
    
    thresh = 0.01
    mod_p = table.filter(pl.col('p-value').lt(thresh))
    coef_heat = mod_p.to_pandas().pivot(index="predictor", columns="target", values="coef")
    sb.heatmap(coef_heat, center=0)
    pyplot.title("univariate (with intercept) model coefficients where p-value < {} (not adjusted)".format(thresh))
    pyplot.show()
    
    at_least_half_gen = table.filter(pl.col('OOS R2') >= (pl.col('train R2') / 2))
    coef_heat = at_least_half_gen.to_pandas().pivot(index="predictor", columns="target", values="coef")
    sb.heatmap(coef_heat, center=0)
    pyplot.title("univariate (with intercept) model coefficients where OOS R2 is at least half of training sample R2")
    pyplot.show()
    
    table.write_csv("univariate_analyses.csv")
    
    # female-specific analyses
    
    fem_rows = main(True)
    table = pl.from_dicts(fem_rows)
    
    mod_p = table.filter(pl.col('p-value').lt(thresh))
    coef_heat = mod_p.to_pandas().pivot(index="predictor", columns="target", values="coef")
    sb.heatmap(coef_heat, center=0)
    pyplot.title("univariate (with intercept) model coefficients where p-value < {} (not adjusted)".format(thresh))
    pyplot.suptitle("female only")
    pyplot.show()  
    
    
    at_least_half_gen = table.filter(pl.col('OOS R2') >= (pl.col('train R2') / 2))
    coef_heat = at_least_half_gen.to_pandas().pivot(index="predictor", columns="target", values="coef")
    sb.heatmap(coef_heat, center=0)
    pyplot.title("univariate (with intercept) model coefficients where OOS R2 is at least half of training sample R2")
    pyplot.suptitle("female only")
    pyplot.show()
    
    table.write_csv("female_univariate_analyses.csv")
    