from whereami import ROOT
import os
from recalibrate.single.singlecalibration import single_calibration
import pandas as pd

WALMART_CSV = ROOT+os.path.sep + 'examples' + os.path.sep + 'walmart_survey' + os.path.sep + 'walmart_survey_clean.csv'
CALIBRATED_CSV = ROOT+os.path.sep + 'examples' + os.path.sep + 'walmart_survey' + os.path.sep + 'walmart_survey_p1_calibrated.csv'


if __name__=='__main__':
    df = pd.read_csv(WALMART_CSV)
    df, report = single_calibration(df, prob_col='p1', new_col='p1_calibrated', y_col='y', by='survey_id')
    df.to_csv(CALIBRATED_CSV)
    print(report)

