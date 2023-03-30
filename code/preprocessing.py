import pandas as pd
import numpy as np

feats_to_be_collapsed = [("new_or_enlarged_lesions_T2", 5, None),
                         ("number_of_new_or_enlarged_lesions_T2",   5, None),
                         ("altered_potential", 9, None),
                         ("potential_value", 9, None),
                         ("delta_relapse_time0", 3, None),
                         ("mri_area_label", 6, None),
                         ("delta_mri_time0", 6, None),
                         ("lesions_T1", 3, None),
                         ("lesions_T2", 3, None),
                         ("delta_evoked_potential_time0", 9, None),
                         ("lesions_T1_gadolinium", 5, None),
                         ("number_of_lesions_T1_gadolinium", 6, None),
                         ("edss_as_evaluated_by_clinician", 11, None),
                         ("location", 9, None),
                         ("delta_edss_time0", 10, None),
                         ("number_of_total_lesions_T2", 3, None)]


def collapse_cols(df, feats_to_be_collapsed):
    for feat in feats_to_be_collapsed:
        df = collapse_ts_feature_cols(df, *feat)
    return df


def collapse_ts_feature_cols(df, feature, start_idx, end_idx=None):
    selected_cols = [col_name for col_name in df.columns.values.tolist() if col_name.startswith(feature)]
    if end_idx:
        cols_to_collapse = [col for col in selected_cols if (start_idx <= int(col[-2:] < end_idx))]
        new_feat_name = f"{feature}_{start_idx}-{end_idx}"
    else:
        cols_to_collapse = [col for col in selected_cols if (start_idx <= int(col[-2:]))]
        new_feat_name = f"{feature}_{start_idx}+"

    df[new_feat_name] = df[cols_to_collapse].isna().all(axis=1)
    df[new_feat_name] = df[new_feat_name].astype(int)
    df = df.drop(cols_to_collapse, axis=1)
    return df


def df_one_hot_encode(original_dataframe, feature_to_encode, drop_org=False):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]]).astype(dtype=np.int64)
    res = pd.concat([original_dataframe, dummies], axis=1)
    if drop_org:
        res = res.drop(feature_to_encode, axis=1)
    return res


def df_ts_func(df, feature, func, func_name):
    calculated_values = []
    for ts in df[feature]:
        if ts is None:
            calculated_values.append(None)
        else:
            calculated_values.append(func(ts))
    out = {f"{feature}_{func_name}": calculated_values}
    return pd.DataFrame(out)


def df_normalize_col(df, feature):
    pass


def preprocess(merged_df):
    supported_funcs = {"len": len, "min": np.min, "max": np.max, "sum": np.sum, "avg": np.average}
    merged_df = merged_df.copy()

    features_to_encode = ["sex", "residence_classification", "ethnicity", 'ms_in_pediatric_age', "centre",
                          'spinal_cord_symptom', 'brainstem_symptom', 'eye_symptom', 'supratentorial_symptom',
                          "other_symptoms"
                          ]
    for feature in features_to_encode:
        merged_df = df_one_hot_encode(merged_df, feature, drop_org=True)

    feature = "edss_as_evaluated_by_clinician"
    created_dfs = [df_ts_func(merged_df, feature, func, name) for name, func in supported_funcs.items()]
    merged_df = pd.concat([merged_df, *created_dfs], axis=1)

    return merged_df