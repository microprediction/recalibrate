from sklearn.metrics import brier_score_loss
from humpday.optimizers.nevergradcube import nevergrad_oneplus_cube
from typing import List
from recalibrate.unarytransforms.alltransforms import get_single_transforms



def calibrate_dataframe_transform(df, transform, prob_col:str, y_col:str, n_trials, n_dim, new_col, by, best_yet=None)->(float,List[float]):
    """
        Find transform parameter r that minimizes brier score
    """

    df_copy = df.copy(deep=True)
    global best_score
    best_score = best_yet

    def objective(r):
        global best_score
        dg = transform(df_copy, r=r, prob_col=prob_col, new_col=new_col, by=by)
        if any(dg[new_col].isna()):
            score = 10000000000
        else:
            try:
                score =  brier_score_loss(dg[y_col], dg[new_col])
            except:
                raise ValueError('problem computing Brier - '+y_col+' may contain bad values??')
        if (best_score is None) or (score<best_score):
            best_score = score
            print({'name':transform.__name__,'brier':best_score,'r':r})
        return score

    best_val, best_r = nevergrad_oneplus_cube(objective, n_trials=n_trials, n_dim=n_dim, with_count=False)
    return best_val, best_r


def single_systematic_calibration(df, prob_col, new_col, y_col, by, include_ability=False, n_trials=50):

    r_best_score_yet = 10000000000
    report = dict()
    for trns, n_dim in get_single_transforms(include_ability=include_ability):
       print('Trying '+trns.__name__)
       r_best_score, r_best = calibrate_dataframe_transform(df=df, transform=trns, prob_col=prob_col, y_col=y_col, n_trials=n_trials, n_dim=n_dim, new_col=new_col, by=by, best_yet=r_best_score_yet)
       if r_best_score < r_best_score_yet:
           r_best_yet = [ri for ri in r_best]
           r_best_score_yet = r_best_score
           report = {'name':trns.__name__,'r_best':r_best_yet,'brier score':r_best_score_yet,'transform':trns}

    df = report['transform'](df=df,prob_col=prob_col, new_col=new_col, r=report['r_best'],by=by)
    return df, report



