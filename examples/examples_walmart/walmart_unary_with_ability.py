import pandas as pd
from recalibrate.unarycalibration.singelsystematiccalibration import single_systematic_calibration
from pprint import pprint
from sklearn.metrics import brier_score_loss
from recalibrate.unarytransforms.normalization import normalize_groups

if __name__=='__main__':

    df_big = pd.read_csv('https://raw.githubusercontent.com/microprediction/recalibrate/main/examples/walmart_survey/walmart_survey_clean.csv')

    if False:
        # Have a guess at a transform?
        df_big['p1'] = df_big['p1'].apply( lambda x: x**1.3 )
        df_big = normalize_groups(df=df_big,by='survey_id',prob_col='p1', new_col='p1')

    n_train = 5000
    train_surveys = set(df_big['survey_id'][:n_train])
    test_surveys = set(df_big['survey_id'][:n_train+100:])
    df_train = df_big[df_big['survey_id'].isin(train_surveys)]
    df_test = df_big[df_big['survey_id'].isin(test_surveys)]

    # Create a new column p1_prime that is similar to probabilities p1
    df_train, report = single_systematic_calibration(df=df_train, prob_col='p1', new_col='p1_prime', y_col='y', by='survey_id', n_trials=25, include_ability=True)
    print('Calibration completed')
    pprint(report)

    # Apply to whole data-frame
    transform = report['transform']
    best_r = report['r_best']
    df_test = transform(df_test, by='survey_id', prob_col='p1', new_col='p1_prime', r=best_r)

    # See if it helps (it does, a little bit)
    briers = {'brier p1':brier_score_loss(df_train['y'], df_train['p1']),
              'brier p1 prime':brier_score_loss(df_test['y'], df_test['p1_prime'])}
    pprint(briers)









