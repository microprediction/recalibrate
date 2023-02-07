from recalibrate.unarytransforms.powertransforms import dataframe_transform_factory
from recalibrate.unarytransforms.activation import ACTIVATIONS, INVERSE_ACTIVATIONS
from recalibrate.unarytransforms.normalization import enormalize


def one_param_activation_transform_factory(a_name, df, by, r, prob_col, new_col):

    def transform(p:[float],r:[float]):
        activation_function = ACTIVATIONS[a_name]
        inverse_activation_function = INVERSE_ACTIVATIONS['ht'] # For now
        return enormalize( [ activation_function( 2*r[0]*inverse_activation_function(pi)) for pi in p ] )

    return dataframe_transform_factory(transform=transform, df=df, by=by, r=r,prob_col=prob_col,new_col=new_col)


def id_transform(df, by, r, prob_col, new_col):
    return one_param_activation_transform_factory(a_name='id', df=df, by=by, r=r, prob_col=prob_col, new_col=new_col)


def pw_transform(df, by, r, prob_col, new_col):
    return one_param_activation_transform_factory(a_name='pw', df=df, by=by, r=r, prob_col=prob_col, new_col=new_col)


def hs_transform(df, by, r, prob_col, new_col):
    return one_param_activation_transform_factory(a_name='hs', df=df, by=by, r=r, prob_col=prob_col, new_col=new_col)


def sg_transform(df, by, r, prob_col, new_col):
    return one_param_activation_transform_factory(a_name='sg', df=df, by=by, r=r, prob_col=prob_col, new_col=new_col)


def bs_transform(df, by, r, prob_col, new_col):
    return one_param_activation_transform_factory(a_name='bs', df=df, by=by, r=r, prob_col=prob_col, new_col=new_col)


def ht_transform(df, by, r, prob_col, new_col):
    return one_param_activation_transform_factory(a_name='ht', df=df, by=by, r=r, prob_col=prob_col, new_col=new_col)


def at_transform(df, by, r, prob_col, new_col):
    return one_param_activation_transform_factory(a_name='at', df=df, by=by, r=r, prob_col=prob_col, new_col=new_col)


def et_transform(df, by, r, prob_col, new_col):
    return one_param_activation_transform_factory(a_name='et', df=df, by=by, r=r, prob_col=prob_col, new_col=new_col)


ONE_PARAM_ACTIVATION_TRANSFORMS = [pw_transform, hs_transform, sg_transform, bs_transform, ht_transform, at_transform, et_transform ]
ONE_PARAM_ACTIVATION_DIMENSIONS = [ 1 for _ in ONE_PARAM_ACTIVATION_TRANSFORMS ]