from whereami import ROOT
import os
from sklearn.metrics import brier_score_loss
from pprint import pprint

WALMART_CSV = ROOT+os.path.sep + 'examples' + os.path.sep + 'walmart_survey' + os.path.sep + 'walmart_survey_clean.csv'
PROBABILITIES_CSV = WALMART_CSV.replace('clean','probabilities')
LOGISTIC_CSV = WALMART_CSV.replace('clean', 'logistic')


def _normalize(p):
    p = [min(max(pi, 0), 1) for pi in p]
    sump = sum(p)
    return [pi / sump for pi in p]


def _normalize_survey(df, prob_col):
    p = df[prob_col]
    df[prob_col] = _normalize(p)
    return df


def normalize_surveys(df, by, prob_col):
    return df.groupby(by).apply(_normalize_survey, prob_col=prob_col)

import math

if __name__=='__main__':
    import pandas as pd
    df = pd.read_csv(LOGISTIC_CSV)


    from recalibrate.unarytransforms.activation import ACTIVATIONS
    for a, f in ACTIVATIONS.items():
        print(a)
        df['p_'+a] = (1.0*(df['log_p1']-df['log_p2']).apply(f) + df['log_p2']).apply(math.exp)
        df = normalize_surveys(df=df,by='survey_id',prob_col='p_'+a)


    briers = dict()
    active_cols = ['p_' + a for a in ACTIVATIONS.keys()]
    for p_name in ['p1', 'p2'] + active_cols:
        briers[p_name] = brier_score_loss(df['y'], df[p_name])


    sorted_briers = sorted([(v,k) for k,v in briers.items()])
    print(' ')
    print('Brier scores')
    pprint(sorted_briers)