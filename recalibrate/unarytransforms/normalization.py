from winning.lattice_calibration import normalize


def enormalize(p):
    try:
        return normalize(p)
    except ZeroDivisionError:
        return [ 1/len(p) for _ in p ]


def normalize_groups(df, by:str, prob_col:str, new_col:str):

    def _normalize_groups(df, prob_col):
        p = df[prob_col]
        df[new_col] = enormalize(p)
        return df

    return df.groupby(by).apply(_normalize_groups, prob_col=prob_col)