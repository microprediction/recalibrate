from recalibrate.unarytransforms.normalization import enormalize
import math


def dataframe_transform_factory(transform, df, by, r, prob_col, new_col):

    def _transform_within_group(df_group,r, prob_col, new_col ):
        p = transform(df_group[prob_col],r=r)
        p = enormalize(p)
        df_group[new_col] = p
        return df_group

    kwargs = {'r':r, 'prob_col':prob_col, 'new_col':new_col }
    df = df.groupby(by=by, group_keys=False).apply( _transform_within_group, **kwargs )
    return df


def _power_up(p:[float], r:[float])->[float]:
    return enormalize([math.pow(pi, r[0]) for pi in p])


def _power_down(p:[float], r:float)->[float]:
    expon = 1/max(r[0],1e-6)
    return enormalize([math.pow(pi, expon) for pi in p])


def power_up(df, by, r, prob_col, new_col):
    return dataframe_transform_factory(df=df, transform=_power_up, by=by, r=r, prob_col=prob_col, new_col=new_col)



def power_down(df, by, r, prob_col, new_col):
    return dataframe_transform_factory(df=df, transform=_power_up, by=by, r=r, prob_col=prob_col, new_col=new_col)


UNARY_POWER_TRANSFORMS = [power_up, power_down]
UNARY_POWER_DIMENSIONS = [1 for _ in UNARY_POWER_TRANSFORMS]