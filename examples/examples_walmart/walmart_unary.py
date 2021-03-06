import pandas as pd
from recalibrate.unarycalibration.singelsystematiccalibration import single_systematic_calibration
from pprint import pprint
from sklearn.metrics import brier_score_loss

# Illustrates calibration of a single set of model probabilities (user selecting a product)

if __name__=='__main__':

    df_big = pd.read_csv('https://raw.githubusercontent.com/microprediction/recalibrate/main/examples/walmart_survey/walmart_survey_clean.csv')

    n_train = 5000
    train_surveys = set(df_big['survey_id'][:n_train])
    test_surveys = set(df_big['survey_id'][:n_train+100:])
    df_train = df_big[df_big['survey_id'].isin(train_surveys)]
    df_test = df_big[df_big['survey_id'].isin(test_surveys)]

    # Create a new column p1_prime that is similar to probabilities p1
    df_train, report = single_systematic_calibration(df=df_train, prob_col='p1', new_col='p1_prime', y_col='y', by='survey_id', n_trials=50, include_ability=False)
    print('Calibration completed')
    pprint(report)

    # Apply to whole data-frame
    transform = report['transform']
    best_r = report['r_best']
    df_test = transform(df_test, by='survey_id', prob_col='p1', new_col='p1_prime', r=best_r)

    # See if it helps (it does, a little bit)
    briers = {'brier before calibration':brier_score_loss(df_test['y'], df_test['p1']),
              'brier after calibration':brier_score_loss(df_test['y'], df_test['p1_prime'])}
    pprint(briers)









