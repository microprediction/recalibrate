import pandas as pd
from recalibrate.unarycalibration.singelsystematiccalibration import single_systematic_calibration
from pprint import pprint
from sklearn.metrics import brier_score_loss
from recalibrate.unarytransforms.normalization import normalize_groups

# Illustrates repeated application of transforms

if __name__=='__main__':

    df_big = pd.read_csv('https://raw.githubusercontent.com/microprediction/recalibrate/main/examples/walmart_survey/walmart_survey_clean.csv')

    n_train = 5000
    train_surveys = set(df_big['survey_id'][:n_train])
    test_surveys = set(df_big['survey_id'][:n_train+100:])
    df_train = df_big[df_big['survey_id'].isin(train_surveys)]
    df_test = df_big[df_big['survey_id'].isin(test_surveys)]

    # Create a new column p1_prime that is similar to probabilities p1 ... and repeat
    sequence = list()
    for i in range(3):
        print('Searching for transform ' + str(i)+' in sequence ')
        df_train, report = single_systematic_calibration(df=df_train, prob_col='p1', new_col='p1_prime', y_col='y', by='survey_id', n_trials=50, include_ability=False)
        pprint(report)
        sequence.append(report)
        df_train['p1'] = df_train['p1_prime']

    print('Here is the sequence of operations')
    pprint(sequence)


    # Apply to whole data-frame
    df_test['p1_copy'] = [pi for pi in df_test['p1'].values]
    for op in sequence:
        transform = op['transform']
        best_r = op['r_best']
        df_test = transform(df_test, by='survey_id', prob_col='p1', new_col='p1', r=best_r)

    # See if it helps (it does, a little bit)
    briers = {'brier after transforms':brier_score_loss(df_test['y'], df_test['p1']),
              'brier before transforms':brier_score_loss(df_test['y'], df_test['p1_copy'])}
    pprint(briers)









