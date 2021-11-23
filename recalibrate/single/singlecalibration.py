from sklearn.metrics import brier_score_loss
from humpday.optimizers.nevergradcube import nevergrad_oneplus_cube
from typing import List


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
        score =  brier_score_loss(dg[y_col], dg[new_col])
        if (best_score is None) or (score<best_score):
            best_score = score
            print({'name':transform.__name__,'brier':best_score,'r':r})
        return score

    best_val, best_r = nevergrad_oneplus_cube(objective, n_trials=n_trials, n_dim=n_dim, with_count=False)
    return best_val, best_r


def single_calibration(df, prob_col, new_col, y_col, by, include_ability=False, n_trials=50 ):
    from recalibrate.transforms.alltransforms import get_single_transforms

    r_best_score = 10000000000
    report = dict()
    for trns, n_dim in get_single_transforms(include_ability=include_ability):
       print('Trying '+trns.__name__)
       prev_best = r_best_score
       r_best_score, r_best = calibrate_dataframe_transform(df=df, transform=trns, prob_col=prob_col, y_col=y_col, n_trials=n_trials, n_dim=n_dim, new_col=new_col, by=by, best_yet=r_best_score)
       if r_best_score < prev_best:
           report = {'name':trns.__name__,'r_best':r_best,'brier score':r_best_score,'transform':trns}

    df = report['transform'](df=df,prob_col=prob_col, new_col=new_col, r=report['r_best'],by=by)
    return df, report



