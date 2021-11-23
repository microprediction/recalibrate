from recalibrate.transforms.powertransforms import SINGLE_POWER_TRANSFORMS, SINGLE_POWER_DIMENSIONS
from recalibrate.transforms.abilitytransforms import SINGLE_ABILITY_TRANSFORMS, SINGLE_ABILITY_DIMENSIONS


def get_single_transforms(include_ability:bool):
    """
        Returns a list of transforms
    :param include_ability:
    :return:
    """
    if include_ability:
        return list(zip(SINGLE_POWER_TRANSFORMS+SINGLE_ABILITY_TRANSFORMS, SINGLE_POWER_DIMENSIONS+SINGLE_ABILITY_DIMENSIONS))
    else:
        return list(zip(SINGLE_POWER_TRANSFORMS, SINGLE_POWER_DIMENSIONS))

