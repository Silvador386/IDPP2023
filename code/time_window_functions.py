import numpy as np
from scipy.stats import mode


def count_not_nan(array):
    not_nan_value_sum = np.sum(~np.isnan(array))
    return not_nan_value_sum


def mode_wrapper(array):
    if np.isnan(array).all():
        return np.nan
    else:
        return mode(array, keepdims=False, nan_policy="omit").mode


def select_max_mri(array):
    pass
#     for cat_type in category_list:
#         if cat_type in array:
#             return np.array(cat_type).astype(str)
#     return np.nan
