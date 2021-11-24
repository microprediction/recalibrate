from recalibrate.unarytransforms.powertransforms import UNARY_POWER_TRANSFORMS, UNARY_POWER_DIMENSIONS
from recalibrate.unarytransforms.abilitytransforms import UNARY_ABILITY_TRANSFORMS, UNARY_ABILITY_DIMENSIONS
from recalibrate.unarytransforms.activationtransforms import ONE_PARAM_ACTIVATION_TRANSFORMS, ONE_PARAM_ACTIVATION_DIMENSIONS


def get_single_transforms(include_ability:bool):
    """
        Returns a list of transforms
    :param include_ability:
    :return:
    """
    transforms = UNARY_POWER_TRANSFORMS + ONE_PARAM_ACTIVATION_TRANSFORMS
    dimensions = UNARY_POWER_DIMENSIONS + ONE_PARAM_ACTIVATION_DIMENSIONS

    if include_ability:
        transforms += UNARY_ABILITY_TRANSFORMS
        dimensions += UNARY_ABILITY_DIMENSIONS

    assert len(transforms)==len(dimensions)
    return list(zip(transforms,dimensions))

