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
    import numpy as np

    from recalibrate.transforms.activation import ACTIVATIONS
    import random
    best_so_far = 0.1
    while True:
        a = random.choice(list(ACTIVATIONS.keys()))
        f = ACTIVATIONS[a]
        c1 = np.random.exponential()*np.random.exponential()
        c2 = np.random.exponential()*np.random.exponential()
        d1 = np.random.rand()-0.5
        d2 = np.random.rand() - 0.5
        sgn = random.choice([1,-1])
        col_name = 'try'
        if sgn<0:
            def g(x):
                return 1-f(-x)
        else:
            g = f

        df[col_name] = (d2*(   (c1*( df['log_p1']-df['log_p2']) - d1).apply(g) + df['log_p2']  )).apply(math.exp)
        df = normalize_surveys(df=df,by='survey_id',prob_col=col_name)
        the_brier = brier_score_loss(df['y'], df[col_name])
        if the_brier<best_so_far:
            pprint({'a':a,
                    'c1':c1,
                    'c2':c2,
                    "d1":d1,
                    "d2":d2,
                    "sgn":sgn,
                    'brier':the_brier})
            best_so_far = the_brier


