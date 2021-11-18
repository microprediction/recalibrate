from recalibrate.inclusion.pandasinclusion import using_pandas
from whereami import ROOT
import os

WALMART_CSV = ROOT+os.path.sep + 'examples' + os.path.sep + 'walmart_survey' + os.path.sep + 'walmart_survey_clean.csv'



if __name__=='__main__':
    assert  using_pandas, 'pip install pandas'
    import pandas as pd
    df = pd.read_csv(WALMART_CSV)
