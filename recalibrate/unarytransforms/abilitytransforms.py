from winning.lattice import skew_normal_density
from winning.pandas_util import add_ability_implied_state_price_to_dataframe
from winning.pandas_util import add_centered_ability_to_dataframe
from typing import Union,List


def ability_expansion(df, by: str, r:List[float], prob_col, ability_col='ability', new_col='prob_new',
                      a = 1.0, scale=1.0, loc=0, unit=0.1, L=251):
    """
        Scale skew-normal abilities by r[0]+0.5, acting within groups
    """
    density = skew_normal_density(L=L, unit=unit, loc=loc, scale=scale, a=a)
    df = add_centered_ability_to_dataframe(df=df, prob_col=prob_col, by=by, density=density, unit=unit, new_col=ability_col)
    expansion = r[0]+0.5
    df[ability_col] = df[ability_col]*expansion
    df = add_ability_implied_state_price_to_dataframe(df=df,ability_col=ability_col, by=by, density=density, new_col=new_col, unit=unit)
    return df




def ability_transform_factory(df, by: str, r:Union[float,List[float]], prob_col, g, ability_col='ability', new_col='prob_new',
                               scale=1.0, loc=0, unit=0.05, L=500):
    """
        More general transform of ability. Here g: (x:float,r:[float])->R is some parametrized transformation
    """
    density = skew_normal_density(L=L, unit=unit, loc=loc, scale=scale, a=a)
    df = add_centered_ability_to_dataframe(df=df, prob_col=prob_col, by=by, density=density, unit=unit, new_col=ability_col)
    def gr(x):
        return g(x,r=r)
    df[ability_col] = df[ability_col].apply(gr)
    df = add_ability_implied_state_price_to_dataframe(df=df,ability_col=ability_col, by=by, density=density, new_col=new_col)
    return df



UNARY_ABILITY_TRANSFORMS = [ability_expansion]
UNARY_ABILITY_DIMENSIONS = [1 for _ in UNARY_ABILITY_TRANSFORMS]
