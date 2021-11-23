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


def _normalize_survey(df_group, p_col):
    p = df_group[p_col]
    df_group[p_col] = _normalize(p)
    return df_group


def normalize_surveys(df, by, p_col):
    return df.groupby(by).apply(_normalize_survey, p_col=p_col)

import math

BACK = True

if __name__=='__main__':
    import pandas as pd
    df = pd.read_csv(LOGISTIC_CSV)


    from recalibrate.transforms.activation import ACTIVATIONS

    f = ACTIVATIONS['bs']
    COEFS = [(a,b) for a in range(0,100,5) for b in [95,100,105,110,115,120]]
    prob_names = list()
    penalty_cols = list()
    unit_cols = list()
    df['d2'] = [1/p for p in df['p2']]
    df = normalize_surveys(df=df, by='survey_id', p_col='p2')
    if any(df['p1']>1):
        raise ValueError('hmmm')
    if any(df['p2'] > 1):
        raise ValueError('hmmm')

    for C1,C2 in COEFS:
        prob_col = 'p_' + str(C1) + '_' + str(C2)
        prob_names.append(prob_col)
        c1 = C1/100.0
        c2 = C2/100.0
        df[prob_col] = ((c1 * (df['log_p1'] - df['log_p2']) + df['log_p2'])*c2).apply(math.exp)
        df = normalize_surveys(df=df, by='survey_id', p_col=prob_col)
        if any(df[prob_col] > 1):
            pass
        # Phrase the problem as adversarial
        penalty_col = prob_col + '_pen'
        penalty_cols.append(penalty_col)
        unit_col = penalty_col.replace('_pen','_unit')
        unit_cols.append(unit_col)
        threshold = 1.2
        if BACK:
            df[unit_col] = ((df[prob_col] > threshold * df['p2']).apply(float) * df[prob_col])
            df[penalty_col] = df[unit_col] * (df['d2']*df['y'] - 1.0)
            pass
        else:
            df[unit_col] = ((threshold * df[prob_col] < df['p2']).apply(float) * df[prob_col])
            df[penalty_col] = df[unit_col] * (1.0 - df['d2']*df['y'])

    briers = dict()
    for p_name in ['p1', 'p2'] + prob_names:
        briers[p_name] = brier_score_loss(df['y'], df[p_name])

    sorted_briers = sorted([(v,k) for k,v in briers.items()])
    print(' ')
    print('Brier scores')
    pprint(sorted_briers[:20])

    the_mean = df[penalty_cols+['survey_id']].mean().reset_index().sort_values(by=0,ascending=False).rename(columns={0:'mu'})
    the_std  = df[penalty_cols+['survey_id']].std().reset_index().sort_values(by=0,ascending=False).rename(columns={0:'std'})
    the_sum = df[['survey_id']+unit_cols].mean().reset_index().sort_values(by=0, ascending=False).rename(columns={0: 'size'}).rename(columns=dict(zip(unit_cols,penalty_cols)))

    def rpl(un):
        return un.replace('_unit','_pen')
    the_sum['index'] = the_sum['index'].apply( rpl )


    the_mean = the_mean.merge(the_std,on='index').merge(the_sum,on='index')
    the_mean['info'] = the_mean['mu']/the_mean['std']
    the_mean['return'] = the_mean['mu'] / the_mean['size']

    the_mean = the_mean.sort_values('info', ascending=False)
    pprint(the_mean[:10])
