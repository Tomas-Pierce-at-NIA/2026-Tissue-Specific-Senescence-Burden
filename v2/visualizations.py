
import numpy as np
import polars as pl
import arviz as az
from matplotlib import pyplot
import seaborn as sb
from adjustText import adjust_text
import json


def get_feature_names():
    with open("out/features_used.json") as feat_hand:
        feat_list = json.load(feat_hand)
    return feat_list


def get_trace():
    return az.from_netcdf("out/trace.netcdf")


def labeled_w(trace, feat_names):
    w_sum = az.summary(trace, var_names=['weights'])
    w_sum['feat_names'] = feat_names
    w_tab = pl.from_pandas(w_sum)
    return w_tab.with_row_index()


def weights_bayes_factors(trace, w_sum):
    space = np.zeros(shape=(len(w_sum),), dtype=float)
    for i, idx in enumerate(w_sum['index']):
        bf = az.bayes_factor(trace.isel(weights_dim_0=idx), 'weights', 0.0)
        nz_favor = bf['BF10']
        space[i] = nz_favor
    return space


def bf_volcano(w_sum):
    fig, ax = pyplot.subplots()
    weights_color = w_sum.with_columns(
        evidence = pl.when(pl.col('BayesFactor').log10() < 1)
                        .then(pl.lit('weak'))
                        .otherwise(
                            pl.when(pl.col('mean') < 0)
                            .then(pl.lit('strong negative association'))
                            .otherwise(pl.lit('strong positive association'))
                        ),
        bf_log10 = pl.col('BayesFactor').log10()
    )
    
    left_5 = weights_color.filter(pl.col('mean') < 0).sort('BayesFactor').tail(5).sort('mean')
    right_5 = weights_color.filter(pl.col('mean') > 0).sort('BayesFactor').tail(5).sort('mean')
    
    palette = {'weak': 'grey', 
               'strong negative association': 'blue', 
               'strong positive association':'red'
               }
    sb.scatterplot(weights_color, x='mean', y='bf_log10', hue='evidence', ax=ax, palette=palette)
    ax.set_xlabel("posterior coefficient mean")
    ax.set_ylabel("Log10 Bayes Factor")
    ax.axhline(0.5, linestyle='--', color='grey')
    ax.axhline(1.0, linestyle='--', color='grey')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0.0, ymax)
    
    text_items = []
    for idx in range(len(left_5)):
        txt = ax.text(x=left_5[idx, "mean"], y=left_5[idx, "bf_log10"], s=str(idx+1))
        text_items.append(txt)
    for idx in range(len(right_5)):
        txt = ax.text(x=right_5[idx, "mean"], y=right_5[idx, "bf_log10"], s=str(5-idx))
        text_items.append(txt)
    
    adjust_text(text_items, time_lim=5, arrowprops=dict(arrowstyle='->', color='black'))
    
    left = left_5.with_row_index('r_idx').select(
        pl.col('feat_names'), pl.col('r_idx')+1, pl.col('mean')
        ).to_numpy()
    right = right_5.with_row_index('r_idx').select(
        pl.col('feat_names'), 5-pl.col('r_idx'), pl.col('mean')
        ).to_numpy()
    labeled = np.concat([left, right], axis=0)
    #left = left_5.with_row_index('r_idx')[:, ['feat_names', 'r_idx']].to_numpy()
    
    #fig.subplots_adjust(hspace=0.2)
    
    fig.tight_layout()
    
    return fig, ax, labeled

def draw_label_table(labeled):
    fig, ax = pyplot.subplots()
    table = ax.table(labeled, 
                         cellLoc='center', 
                         rowLoc='center', 
                         colLoc='center', 
                         loc='center',
                         colWidths=[0.8, 0.1, 0.1],
                         colLabels=["name", "label", "mean"]
                         )
    fig.tight_layout()

if __name__ == '__main__':
    trace = get_trace()
    features = get_feature_names()
    labeled_weight_summary = labeled_w(trace, features)
    bayes_factors = weights_bayes_factors(trace, labeled_weight_summary)
    bf = pl.Series("BayesFactor", bayes_factors)
    labeled_weight_summary = labeled_weight_summary.with_columns(bf)
    
    _fig, _ax, labeled = bf_volcano(labeled_weight_summary)
    draw_label_table(labeled)
    pyplot.show()
    
    az.plot_energy(trace)
    pyplot.show()
    
    az.plot_forest(trace, var_names=['icpt'], hdi_prob=0.9)
    pyplot.show()
    
    az.plot_ppc(trace, num_pp_samples=100)
    pyplot.show()
    
    az.plot_loo_pit(trace, 'y')
    pyplot.show()
    
    az.plot_bpv(trace, kind='p_value')
    pyplot.show()
    
    az.plot_bf(trace.isel(weights_dim_0=99), 'weights', ref_val=0.0)
    pyplot.show()
    
    r2 = az.r2_score(trace.predictions_constant_data['ydata'].values, 
                trace.predictions.stack(sample=('chain','draw'))['y'].T.values
                )
    
    print(r2)
    
    