from recalibrate.inclusion.pandasinclusion import using_pandas
from whereami import ROOT
import os
from pprint import pprint
from winning.pandas_util import add_skew_normal_ability_to_dataframe
from winning.lattice_calibration import ability_implied_state_prices, state_price_implied_ability
from winning.lattice import skew_normal_density
from winning.lattice_conventions import STD_A,STD_L, STD_SCALE, STD_UNIT
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression

def center(x):
    mx = sum(x) / len(x)
    return [xi - mx for xi in x]

WALMART_CSV = ROOT+os.path.sep + 'examples' + os.path.sep + 'walmart_survey' + os.path.sep + 'walmart_survey_clean.csv'
ABILITIES_CSV = WALMART_CSV.replace('clean','ability')
PROBABILITIES_CSV = ABILITIES_CSV.replace('ability', 'probabilities')
LOGISTIC_CSV = ABILITIES_CSV.replace('ability', 'logistic')


if using_pandas:

    def _normalize(p):
        p = [ min(max(pi,0),1) for pi in p]
        sump = sum(p)
        return [pi/sump for pi in p]


    def _normalize_survey(df,prob_col):
        p = df[prob_col]
        df[prob_col] = _normalize(p)
        return df

    def normalize_surveys(df,by,prob_col):
        return df.groupby(by).apply(_normalize_survey,prob_col=prob_col)

    def add_ability_implied_probability_to_dataframe(df, ability_col, by:str, density, new_col:str, c1, c2, attractor_col):

        def _add_prob(df, ability_col, new_col, density, c1,c2, attractor_col):
            abilities = df[ability_col].values
            if attractor_col is None:
                shrunk_ability = [ ai*c1 for ai in abilities ]
            else:
                attractors = df[attractor_col].values
                shrunk_ability = [ c1*ai + c2*bi for ai,bi in zip(abilities,attractors) ]
            p = ability_implied_state_prices(ability=shrunk_ability, density=density, unit=STD_UNIT )
            a_check = center( state_price_implied_ability(prices=p, density=density, unit=STD_UNIT) )
            p = _normalize(p)
            df[new_col] = p
            if any([pi<0 for pi in p]):
                raise Exception('huh')
            return df

        kwargs = {'ability_col': ability_col, 'new_col': new_col, 'density': density, 'c1':c1,'c2':c2, 'attractor_col':attractor_col}
        return df.groupby(by).apply(_add_prob, **kwargs)

    def add_skew_normal_probability_to_dataframe(df, by: str, ability_col='a', new_col='ability', L=STD_L, scale=STD_SCALE,
                                                 unit=STD_UNIT, a=STD_A, loc=0.0, c1=1.0, c2=0.0, attractor_col=None):
        density = skew_normal_density(L=L, unit=unit, loc=loc, scale=scale, a=a)
        return add_ability_implied_probability_to_dataframe(df=df, ability_col=ability_col, by=by, new_col=new_col, density=density,
                                                            c1=c1, c2=c2, attractor_col=attractor_col)


if __name__=='__main__':
    assert  using_pandas, 'pip install pandas'
    import pandas as pd

    n = 5000  # total number of survey responses
    unit = 0.1*STD_UNIT
    L = 2*STD_L

    try:
        df_small = pd.read_csv(ABILITIES_CSV)
        print('Read from '+ABILITIES_CSV)
    except FileNotFoundError:
        df = pd.read_csv(WALMART_CSV)
        some_surveys = list(set( df['survey_id'].values[:n] ))
        df_small = df[df['survey_id'].isin(some_surveys)]

        print('Normalizing')
        df_small = normalize_surveys(df=df_small,by='survey_id',prob_col='p1')
        df_small = normalize_surveys(df=df_small, by='survey_id', prob_col='p2')

        print('Calculating abilities')
        df_small = add_skew_normal_ability_to_dataframe(df=df_small, by='survey_id', prob_col='p1', new_col='a1', a=0, unit=unit)
        df_small = add_skew_normal_ability_to_dataframe(df=df_small, by='survey_id', prob_col='p2', new_col='a2', a=0, unit=unit)

        df_small.to_csv(ABILITIES_CSV)
        print('Wrote '+ABILITIES_CSV)

    print('Adding shrunk probabilities')
    COEFS = [(75,25),(25,75),(64,36),(36,64),(50,50)] + [(70,25),(25,70),(59,36),(36,59),(45,45)] + [(85,25),(25,85),(74,36),(36,74),(55,55)] + [(0,110),(110,0),(0,120),(120,0)]
    coef_cols = list()
    for coef1, coef2 in COEFS:
        new_col  = 'p'+str(coef1)+'_'+str(coef2)
        print(new_col)
        coef_cols.append(new_col)
        df_small = add_skew_normal_probability_to_dataframe(df=df_small, by='survey_id', ability_col='a1', new_col=new_col, c1=0.01*coef1, c2=0.01*coef2, attractor_col=None)
    pprint(df_small)

    # Compare against existing Brier scores
    briers = dict()

    for p_name in ['p1','p2'] + coef_cols:
        briers[p_name]=brier_score_loss(df_small['y'], df_small[p_name])

    sorted_briers = sorted([(v,k) for k,v in briers.items()])
    pprint(sorted_briers)

    df_small.to_csv(PROBABILITIES_CSV)


    print('Performing logistic regression')
    n_train = int(len(df_small)/2)

    import math
    import numpy as np

    p_feature_cols = ['p1', 'p2'] + coef_cols
    feature_cols = ['log_'+fc for fc in p_feature_cols]
    for pfc,fc in zip(p_feature_cols,feature_cols):
        df_small[fc] = df_small[pfc].apply( math.log )
    df_train = df_small[:n_train]  # TODO: Don't split a survey in two
    df_test = df_small[n_train:]

    X = np.array( df_small[feature_cols] )
    X_test = np.array( df_test[feature_cols] )
    X_train = np.array(df_train[feature_cols])
    y_train = df_train['y'].values
    y_test = df_test['y'].values

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    valuable = sorted(list(zip(list(clf.coef_[0]), feature_cols)), key=lambda x: abs(x[0]), reverse=True)
    print('Logistic regression coefs...')
    pprint(valuable)

    df_small['p_mix'] = [ x[1] for x in clf.predict_proba(X) ]
    df_small = normalize_surveys(df=df_small,by='survey_id',prob_col='p_mix')
    df_small_test = df_small[n_train:]
    df_small_train = df_small[:n_train]

    briers['p_mix']=brier_score_loss(df_small_test['y'], df_small_test['p_mix'])
    briers['p_mix_in'] = brier_score_loss(df_small_train['y'], df_small_train['p_mix'])

    sorted_briers = sorted([(v,k) for k,v in briers.items()])
    print(' ')
    print('Brier scores')
    pprint(sorted_briers)
    df_small.to_csv(LOGISTIC_CSV)




