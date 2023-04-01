import pandas as pd
import numpy as np
from fastai.tabular.all import RandomSplitter, range_of, TabularPandas,\
    Categorify, FillMissing, Normalize, CategoryBlock


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


def preprocess(merged_df):
    supported_funcs = {"len": len, "min": np.min, "max": np.max, "sum": np.sum, "avg": np.average}
    merged_df = merged_df.copy()

    features_to_encode = ["sex", "residence_classification", "ethnicity", 'ms_in_pediatric_age', "centre",
                          'spinal_cord_symptom', 'brainstem_symptom', 'eye_symptom', 'supratentorial_symptom',
                          "other_symptoms"
                          ]
    # for feature in features_to_encode:
    #     merged_df = df_one_hot_encode(merged_df, feature, drop_org=True)

    # feature = "edss_as_evaluated_by_clinician"
    # created_dfs = [df_ts_func(merged_df, feature, func, name) for name, func in supported_funcs.items()]
    # merged_df = pd.concat([merged_df, *created_dfs], axis=1)

    merged_df = collapse_cols(merged_df, feats_to_be_collapsed)

    return merged_df


def fastai_splits(df):
    col_value_types = df.columns.to_series().groupby(df.dtypes).groups

    # TODO add option to add outcome_time/outcome_occurred_depending on the preceding calculation

    col_value_types = {"bool": ['ms_in_pediatric_age', 'spinal_cord_symptom', 'brainstem_symptom', 'eye_symptom',
                                'supratentorial_symptom'],
                       "int32": ['new_or_enlarged_lesions_T2_5+', 'number_of_new_or_enlarged_lesions_T2_5+',
                                 'altered_potential_9+', 'potential_value_9+', 'delta_relapse_time0_3+', 'mri_area_label_6+',
                                 'delta_mri_time0_6+', 'lesions_T1_3+', 'lesions_T2_3+', 'delta_evoked_potential_time0_9+',
                                 'lesions_T1_gadolinium_5+', 'number_of_lesions_T1_gadolinium_6+',
                                 'edss_as_evaluated_by_clinician_11+', 'location_9+', 'delta_edss_time0_10+',
                                 'number_of_total_lesions_T2_3+'],
                       "int64": ['age_at_onset', 'time_since_onset'],
                       "float64": ['diagnostic_delay', 'delta_relapse_time0_01', 'delta_relapse_time0_02',
                                   'delta_observation_time0_01', 'delta_observation_time0_02',
                                   'number_of_lesions_T1_gadolinium_01', 'number_of_lesions_T1_gadolinium_02',
                                   'number_of_lesions_T1_gadolinium_03', 'number_of_lesions_T1_gadolinium_04',
                                   'number_of_lesions_T1_gadolinium_05', 'number_of_new_or_enlarged_lesions_T2_01',
                                   'number_of_new_or_enlarged_lesions_T2_02', 'number_of_new_or_enlarged_lesions_T2_03',
                                   'number_of_new_or_enlarged_lesions_T2_04', 'delta_mri_time0_01', 'delta_mri_time0_02',
                                   'delta_mri_time0_03', 'delta_mri_time0_04', 'delta_mri_time0_05',
                                   'delta_evoked_potential_time0_01', 'delta_evoked_potential_time0_02',
                                   'delta_evoked_potential_time0_03', 'delta_evoked_potential_time0_04',
                                   'delta_evoked_potential_time0_05', 'delta_evoked_potential_time0_06',
                                   'delta_evoked_potential_time0_07', 'delta_evoked_potential_time0_08',
                                   'edss_as_evaluated_by_clinician_01', 'edss_as_evaluated_by_clinician_02',
                                   'edss_as_evaluated_by_clinician_03', 'edss_as_evaluated_by_clinician_04',
                                   'edss_as_evaluated_by_clinician_05', 'edss_as_evaluated_by_clinician_06',
                                   'edss_as_evaluated_by_clinician_07', 'edss_as_evaluated_by_clinician_08',
                                   'edss_as_evaluated_by_clinician_09', 'edss_as_evaluated_by_clinician_10',
                                   'delta_edss_time0_01', 'delta_edss_time0_02', 'delta_edss_time0_03',
                                   'delta_edss_time0_04', 'delta_edss_time0_05', 'delta_edss_time0_06',
                                   'delta_edss_time0_07', 'delta_edss_time0_08', 'delta_edss_time0_09'],
                       "object": ['sex', 'residence_classification', 'ethnicity', 'other_symptoms', 'centre',
                                  'multiple_sclerosis_type_01', 'multiple_sclerosis_type_02', 'mri_area_label_01',
                                  'mri_area_label_02', 'mri_area_label_03', 'mri_area_label_04', 'mri_area_label_05',
                                  'lesions_T1_01', 'lesions_T1_02', 'lesions_T1_gadolinium_01', 'lesions_T1_gadolinium_02',
                                  'new_or_enlarged_lesions_T2_01', 'new_or_enlarged_lesions_T2_02',
                                  'new_or_enlarged_lesions_T2_03', 'new_or_enlarged_lesions_T2_04', 'lesions_T2_01',
                                  'lesions_T2_02', 'number_of_total_lesions_T2_01', 'number_of_total_lesions_T2_02',
                                  'altered_potential_01', 'altered_potential_02', 'altered_potential_03',
                                  'altered_potential_04', 'altered_potential_05', 'altered_potential_06',
                                  'altered_potential_07', 'altered_potential_08', 'potential_value_01', 'potential_value_02',
                                  'potential_value_03', 'potential_value_04', 'potential_value_05', 'potential_value_06',
                                  'potential_value_07', 'potential_value_08', 'location_01', 'location_02', 'location_03',
                                  'location_04', 'location_05', 'location_06', 'location_07', 'location_08']
                       }

    cat_names = [*col_value_types["bool"]]
    cont_names = [ *col_value_types["int64"]]

    # cont_names = cont_names.copy().remove(remove_feature)

    splits = RandomSplitter(valid_pct=0.2)(range_of(df))
    return cat_names, cont_names, splits


def fastai_tab(df, cat_names, cont_names, y_names, splits):
    to = TabularPandas(df, procs=[Categorify, FillMissing],
                       cat_names=cat_names,
                       cont_names=cont_names,
                       y_names=("outcome_occurred", "outcome_time"),
                       splits=splits)

    X_train, y_train = to.train.xs, to.train.ys.values.ravel()
    X_valid, y_valid = to.valid.xs, to.valid.ys.values.ravel()

    # y_train = pd.DataFrame({"outcome_occurred": y_train[::2],
    #                         "outcome_time": y_train[1::2]}).to_records()
    #
    # y_valid = pd.DataFrame({"outcome_occurred": y_valid[::2],
    #                         "outcome_time": y_valid[1::2]}).to_records()


    # oo = y_train[::2].astype(dtype=bool)
    # ot = y_train[1::2]
    # y_train = np.array([oo, ot])
    #
    # oo = y_train[::2].astype(dtype=bool)
    # ot = y_train[1::2]
    # y_train = np.array([oo, ot])

    y_train = np.array(
        [(bool(outcome_occurred), outcome_time) for outcome_occurred, outcome_time in zip(y_train[::2], y_train[1::2])],
        dtype=[('outcome_occurred', '?'), ('outcome_time', '<f8')])

    y_valid = np.array(
        [(bool(outcome_occurred), outcome_time) for outcome_occurred, outcome_time in zip(y_valid[::2], y_valid[1::2])],
        dtype=[('outcome_occurred', '?'), ('outcome_time', '<f8')])

    return X_train, y_train, X_valid, y_valid


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

    df[new_feat_name] = ~df[cols_to_collapse].isna().all(axis=1)
    df[new_feat_name] = df[new_feat_name].astype(np.int64)
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
