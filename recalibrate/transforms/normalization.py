import numpy as np


def normalize(p:[float])->float:
    sump = sum(p)
    if sump>0:
        return [pi/sump for pi in p]
    else:
        return p


def normalize_groups(df, by:str, prob_col:str, new_col:str):
    def _normalize_groups(df, prob_col):
        p = df[prob_col]
        df[new_col] = normalize(p)
        return df
    return df.groupby(by).apply(_normalize_groups, prob_col=prob_col)