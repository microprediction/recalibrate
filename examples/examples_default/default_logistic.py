import pandas as pd
from recalibrate.unarycalibration.singelsystematiccalibration import single_systematic_calibration
from pprint import pprint
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier

# Illustrates calibration of a single set of model probabilities (user selecting a product)

if __name__=='__main__':

    df_big = pd.read_csv('https://raw.githubusercontent.com/microprediction/recalibrate/main/examples/default_data/default.csv')[1:].astype(float)

    # We have only one group and multiple 1's ... so can't use ability transforms which would be slow anyway.
    df_big['group_id'] = ['one' for _ in df_big.index]

    n_train = 15000
    df_train = df_big[1:n_train]
    features = [ c for c in df_train.columns if 'X' in c ]
    df_test = df_big[n_train:]


    X_train = df_train[features].values
    y_train = df_train['Y']
    model = XGBClassifier()
    model.fit(X_train, y_train)
    X_hat = model.predict_proba(df_big[features].values)

    df_big['p1'] = [x[1] for x in X_hat ]
    df_train = df_big[:n_train]
    df_test = df_big[n_train:]

    print('Result of xgboost classification ')
    original_brier = brier_score_loss(df_test['Y'], df_test['p1'])
    print( df_train['p1'].describe() )
    print(' xgboost Brier score on test set is '+str(original_brier))

    # Try to recalibrate
    # We have to massage into group format
    # (Thus should be a utility)

    df_train_copy_a = df_train.copy(deep=True)
    df_train_copy_b = df_train.copy(deep=True)
    df_train_copy_b['p1'] = df_train_copy_b['p1'].apply(lambda p: 1-p)
    group_ids = list(range(len(df_train_copy_a)))
    df_train_copy_a['group_id'] = group_ids
    df_train_copy_b['group_id'] = group_ids
    df_train_copy_b['Y'] = df_train_copy_b['Y'].apply(lambda y: 1 - y)
    df_train_big = pd.concat([df_train_copy_a,df_train_copy_b], ignore_index=True )

    df_train, report = single_systematic_calibration(df=df_train_big, prob_col='p1', new_col='p1_prime', y_col='Y', by='group_id', n_trials=50, include_ability=False)
    print('Calibration completed')
    pprint(report)

    # Apply to whole data-frame
    transform = report['transform']
    best_r = report['r_best']

    df_test_copy_a = df_train.copy(deep=True)
    df_test_copy_b = df_train.copy(deep=True)
    df_test_copy_b['p1'] = df_test_copy_b['p1'].apply(lambda p: 1 - p)
    df_test_copy_a['group_id'] = df_test_copy_a['Y']
    df_test_copy_b['group_id'] = df_test_copy_b['Y']
    df_test_copy_b['Y'] = df_test_copy_b['Y'].apply(lambda y: 1 - y)
    df_test_groups = pd.concat([df_test_copy_a, df_test_copy_b], ignore_index=True)
    df_test_groups = transform(df_test_groups, by='group_id', prob_col='p1', new_col='p1_prime', r=best_r)

    # See if it helps (it does, a little bit)
    briers = {'brier of xgboost':brier_score_loss(df_test_groups['Y'], df_test_groups['p1']),
              'brier of recalibrated xgboost':brier_score_loss(df_test_groups['Y'], df_test_groups['p1_prime'])}
    pprint(briers)









