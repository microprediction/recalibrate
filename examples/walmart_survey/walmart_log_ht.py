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


    from recalibrate.transforms.activation import ACTIVATIONS

    f = ACTIVATIONS['ht']
    COEFS = [(a,b) for a in range(100,500,5) for b in range(5,40,3)]
    names = list()
    for C1,C2 in COEFS:
        col_name = 'p_'+str(C1)+'_'+str(C2)
        names.append(col_name)
        c1 = C1/100.0
        c2 = C2/100.0
        df[col_name] = (c1*(c2*(df['log_p1']-df['log_p2'])).apply(f) + df['log_p2']).apply(math.exp)
        df = normalize_surveys(df=df,by='survey_id',prob_col=col_name)

    briers = dict()
    for p_name in ['p1', 'p2'] + names:
        briers[p_name] = brier_score_loss(df['y'], df[p_name])


    sorted_briers = sorted([(v,k) for k,v in briers.items()])
    print(' ')
    print('Brier scores')
    pprint(sorted_briers[:20])