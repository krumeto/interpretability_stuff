import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.inspection import plot_partial_dependence, permutation_importance

import joblib

def load_pdp_model(payer):
    return joblib.load(f'{payer}_pdp.joblib')

def load_pdp_X(payer):
    return pd.read_parquet(f'{payer}_X.parquet')

def load_pdp_y(payer):
    return pd.read_parquet(f'{payer}_y.parquet')

def plot_1000_reimb(clf, X, lim=1000):
    fig, ax = plt.subplots(figsize=(12, 6))
  
    exp_r = plot_partial_dependence(clf, X, ['exp_reimbursement'],subsample=10,
      kind='both', ax=ax,
                                       line_kw={"color": "red"})
    
    exp_r.axes_[0][0].set_xlim([0, lim])
    plt.show()
    
def plot_importance(clf, X, y, top=0):
    result = permutation_importance(clf, X, y, n_repeats=10,
                                    random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    
    if top > 0:
        sorted_idx = sorted_idx[-top:]
  
    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=X.columns[sorted_idx])
    ax.set_title("Permutation Importances")
    fig.tight_layout()
    plt.show()
    
def get_perm_importances(PAYER):
    result = permutation_importance(load_pdp_model(PAYER), load_pdp_X(PAYER), load_pdp_y(PAYER), n_repeats=5,
                                    random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    
    df_alt = pd.DataFrame(result.importances[sorted_idx[-10:]].T, columns=load_pdp_X(PAYER).columns[sorted_idx[-10:]])
    
    return df_alt

def get_all_perm_importances(PAYER):
    result = permutation_importance(load_pdp_model(PAYER), load_pdp_X(PAYER), load_pdp_y(PAYER), n_repeats=5,
                                    random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    
    df_alt = pd.DataFrame(result.importances[sorted_idx].T, columns=load_pdp_X(PAYER).columns[sorted_idx])
    
    return df_alt


def plot_binary_pdps(feature, kind='average'):
    """Kind must be one of ["average", "individual", "both"]"""
    
    fig, ax = plt.subplots(7,2, figsize=(15, 30))

    for ix, PAYER in enumerate(PAYERS):
        
        try:
            ax[ix//2, ix*13%2].set_title(f'{PAYER} - Partial Dependency of {feature}', fontsize=20)
            pdp_plot = plot_partial_dependence(load_pdp_model(PAYER), 
                                               load_pdp_X(PAYER), 
                                               [feature], 
                                               ax=ax[ix//2, ix*13%2],
                                               kind = kind
                                              )
            pdp_plot.axes_[0][0].xaxis.label.set_size(10)
            pdp_plot.axes_[0][0].yaxis.label.set_size(10)
            pdp_plot.axes_[0][0].tick_params(axis = "both", labelsize=8)

            pdp_val = pd.DataFrame({'values':pdp_plot.pd_results[0]['values'][0],
                     'average':pdp_plot.pd_results[0]['average'][0],
                     })

            y_min = pdp_val.average.min()
            y_max = pdp_val.average.max()

            if pdp_val.iloc[0].average < pdp_val.iloc[-1].average:
                impact = f'Increase of Dirty Probability by {inv_logit(y_max) - inv_logit(y_min):.2%}'
            else:
                impact = f'Decrease of Dirty Probability by {inv_logit(y_min) - inv_logit(y_max):.2%}'

            pdp_plot.axes_[0][0].axhline(y=y_max,color='red', linestyle = '-.')
            pdp_plot.axes_[0][0].text(0.5, y_max+0.01, f'{inv_logit(y_max):.2%} probability of Dirty', c='red')
            pdp_plot.axes_[0][0].axhline(y=y_min,color='green', linestyle = '-.')
            pdp_plot.axes_[0][0].text(0.5, y_min+0.01, f'{inv_logit(y_min):.2%} probability of Dirty', c='green')

            pdp_plot.axes_[0][0].text(0.3, y_max+0.15, impact)
            pdp_plot.axes_[0][0].set_ylim([y_min-0.2,y_max+0.2])
            
        except:
            continue

    plt.tight_layout()
    plt.show()
    
def plot_combo_pdp(features, kind='average'):
    
    fig, ax = plt.subplots(7,2, figsize=(20, 40))

    for ix, PAYER in enumerate(PAYERS):

        ax[ix//2, ix*13%2].set_title(f'{PAYER} - PDP of {features[0]} with {features[1]}', fontsize=20)
        pdp_plot = plot_partial_dependence(load_pdp_model(PAYER), 
                                           load_pdp_X(PAYER), 
                                           [features], 
                                           ax=ax[ix//2, ix*13%2], 
                                           kind=kind
                                          )
        pdp_plot.axes_[0][0].xaxis.label.set_size(12)
        pdp_plot.axes_[0][0].yaxis.label.set_size(12)
        pdp_plot.axes_[0][0].tick_params(axis = "both", labelsize=10)

    plt.tight_layout()
    plt.show()
    
def plot_continuous_pdp(feature, kind='average', limit=None):
    """Kind must be one of ["average", "individual", "both"]"""
    fig, ax = plt.subplots(7,2, figsize=(20, 40))

    for ix, PAYER in enumerate(PAYERS):
        if limit:
            ax[ix//2, ix*13%2].set_title(f'{PAYER} - PDP of {feature}, First {limit} USD', fontsize=20)
        else:
            ax[ix//2, ix*13%2].set_title(f'{PAYER} - Partial Dependency of {feature}', fontsize=20)
        pdp_plot = plot_partial_dependence(load_pdp_model(PAYER), 
                                           load_pdp_X(PAYER), 
                                           [feature], 
                                           ax=ax[ix//2, ix*13%2], 
                                           kind=kind
                                          )
        pdp_plot.axes_[0][0].xaxis.label.set_size(12)
        pdp_plot.axes_[0][0].yaxis.label.set_size(12)
        pdp_plot.axes_[0][0].tick_params(axis = "both", labelsize=10)
        
        if limit:
            pdp_plot.axes_[0][0].set_xlim([0, limit])

        #Line Annotations
        pdp_plot.axes_[0][0].axhline(color='blue', linestyle = '-.')
        pdp_plot.axes_[0][0].text(0.5, 0.01, '50.00% probability of Dirty', c='blue')

        if limit:
            pdp_val = pd.DataFrame({'values':pdp_plot.pd_results[0]['values'][0],
                     'average':pdp_plot.pd_results[0]['average'][0],
                     }).loc[lambda df: df['values'] <= limit]
        else:
            pdp_val = pd.DataFrame({'values':pdp_plot.pd_results[0]['values'][0],
                     'average':pdp_plot.pd_results[0]['average'][0],
                     })

        y_min = pdp_val.average.min()
        y_max = pdp_val.average.max()

        y_bounds = pdp_plot.axes_[0][0].get_ybound()
        text_loc = (y_bounds[1]-y_bounds[0])/14
        pdp_plot.axes_[0][0].axhline(y=y_max-0.05,color='red', linestyle = '-.')
        pdp_plot.axes_[0][0].text(0.5, y_max-0.04, f'{inv_logit(y_max-0.05):.2%} probability of Dirty', c='red')
        pdp_plot.axes_[0][0].axhline(y=y_min+0.05,color='green', linestyle = '-.')
        pdp_plot.axes_[0][0].text(0.5, y_min+text_loc, f'{inv_logit(y_min+0.05):.2%} probability of Dirty', c='green')

    plt.tight_layout()
    plt.show()
    
def plot_specific_payer(payer, feature, kind='average', limit=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if limit:
        ax.set_title(f'{payer} - PDP of {feature}, First {limit} USD', fontsize=20)
    else:
        ax.set_title(f'{payer} - Partial Dependency of {feature}', fontsize=20)
    pdp_plot = plot_partial_dependence(load_pdp_model(payer), 
                                           load_pdp_X(payer), 
                                           [feature], 
                                           ax=ax, 
                                           kind=kind
                                          )
    pdp_plot.axes_[0][0].xaxis.label.set_size(12)
    pdp_plot.axes_[0][0].yaxis.label.set_size(12)
    pdp_plot.axes_[0][0].tick_params(axis = "both", labelsize=10)
        
    if limit:
        pdp_plot.axes_[0][0].set_xlim([0, limit])

    #Line Annotations
    pdp_plot.axes_[0][0].axhline(color='blue', linestyle = '-.')
    pdp_plot.axes_[0][0].text(0.5, 0.01, '50.00% probability of Dirty', c='blue')

    if limit:
        pdp_val = pd.DataFrame({'values':pdp_plot.pd_results[0]['values'][0],
                     'average':pdp_plot.pd_results[0]['average'][0],
                     }).loc[lambda df: df['values'] <= limit]
    else:
        pdp_val = pd.DataFrame({'values':pdp_plot.pd_results[0]['values'][0],
                     'average':pdp_plot.pd_results[0]['average'][0],
                     })

    y_min = pdp_val.average.min()
    y_max = pdp_val.average.max()

    y_bounds = pdp_plot.axes_[0][0].get_ybound()
    text_loc = (y_bounds[1]-y_bounds[0])/20
    pdp_plot.axes_[0][0].axhline(y=y_max-text_loc,color='red', linestyle = '-.')
    pdp_plot.axes_[0][0].text(0.5, y_max-text_loc, f'{inv_logit(y_max-text_loc):.2%} probability of Dirty', c='red')
    pdp_plot.axes_[0][0].axhline(y=y_min+text_loc,color='green', linestyle = '-.')
    pdp_plot.axes_[0][0].text(0.5, y_min+text_loc, f'{inv_logit(y_min+text_loc):.2%} probability of Dirty', c='green')

    plt.tight_layout()
    plt.show()