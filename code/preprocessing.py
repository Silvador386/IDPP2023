import pandas as pd
import numpy as np
from fastai.tabular.all import RandomSplitter, range_of, TabularPandas,\
    Categorify, FillMissing, Normalize, CategoryBlock


ALL_FEATURES = ['patient_id', 'sex', 'residence_classification', 'ethnicity',
                'ms_in_pediatric_age', 'age_at_onset', 'diagnostic_delay',
                'spinal_cord_symptom', 'brainstem_symptom', 'eye_symptom',
                'supratentorial_symptom', 'other_symptoms', 'centre',
                'time_since_onset', 'outcome_occurred', 'outcome_time',
                'edss_as_evaluated_by_clinician', 'delta_edss_time0',
                'delta_relapse_time0', 'multiple_sclerosis_type',
                'delta_observation_time0', 'altered_potential', 'potential_value',
                'location', 'delta_evoked_potential_time0', 'mri_area_label',
                'lesions_T1', 'lesions_T1_gadolinium',
                'number_of_lesions_T1_gadolinium', 'new_or_enlarged_lesions_T2',
                'number_of_new_or_enlarged_lesions_T2', 'lesions_T2',
                'number_of_total_lesions_T2', 'delta_mri_time0']

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

    # ts_features = ["age_at_onset", "edss_as_evaluated_by_clinician""delta_edss_time0", "potential_value",
    #                "delta_evoked_potential_time0", "delta_relapse_time0"]
    #
    # merged_df = create_ts_features(merged_df, ts_features, drop_original=True)
    #
    # target_features = ['outcome_occurred', 'outcome_time']
    # unfinished_features = list(set(ALL_FEATURES).difference([*features_to_encode, *ts_features, *target_features]))
    # cols_to_drop = []
    # for un_feat in unfinished_features:
    #     cols_to_drop += select_same_feature_col_names(merged_df, un_feat)
    # merged_df = merged_df.drop(cols_to_drop, axis=1)
    merged_df = collapse_cols(merged_df, feats_to_be_collapsed)

    return merged_df


def fastai_ccnames_original(df, valid_pct=0.2):
    col_value_types = {"bool": ['ms_in_pediatric_age', 'spinal_cord_symptom', 'brainstem_symptom', 'eye_symptom',
                                'supratentorial_symptom'],
                       "int32": ['new_or_enlarged_lesions_T2_5+', 'number_of_new_or_enlarged_lesions_T2_5+',
                                 'altered_potential_9+', 'potential_value_9+', 'delta_relapse_time0_3+',
                                 'mri_area_label_6+',
                                 'delta_mri_time0_6+', 'lesions_T1_3+', 'lesions_T2_3+',
                                 'delta_evoked_potential_time0_9+',
                                 'lesions_T1_gadolinium_5+', 'number_of_lesions_T1_gadolinium_6+',
                                 'edss_as_evaluated_by_clinician_11+',
                                 'location_9+',
                                 'delta_edss_time0_10+',
                                 'number_of_total_lesions_T2_3+'],
                       "int64": ['age_at_onset', 'time_since_onset'],
                       "float64": ['diagnostic_delay', 'delta_relapse_time0_01', 'delta_relapse_time0_02',
                                   'delta_observation_time0_01', 'delta_observation_time0_02',
                                   'number_of_lesions_T1_gadolinium_01', 'number_of_lesions_T1_gadolinium_02',
                                   'number_of_lesions_T1_gadolinium_03', 'number_of_lesions_T1_gadolinium_04',
                                   'number_of_lesions_T1_gadolinium_05', 'number_of_new_or_enlarged_lesions_T2_01',
                                   'number_of_new_or_enlarged_lesions_T2_02', 'number_of_new_or_enlarged_lesions_T2_03',
                                   'number_of_new_or_enlarged_lesions_T2_04', 'delta_mri_time0_01',
                                   'delta_mri_time0_02',
                                   'delta_mri_time0_03', 'delta_mri_time0_04', 'delta_mri_time0_05',
                                   'delta_evoked_potential_time0_01', 'delta_evoked_potential_time0_02',
                                   'delta_evoked_potential_time0_03', 'delta_evoked_potential_time0_04',
                                   'delta_evoked_potential_time0_05', 'delta_evoked_potential_time0_06',
                                   'delta_evoked_potential_time0_07', 'delta_evoked_potential_time0_08',
                                   'edss_as_evaluated_by_clinician_01',
                                   'edss_as_evaluated_by_clinician_02',
                                   'edss_as_evaluated_by_clinician_03', 'edss_as_evaluated_by_clinician_04',
                                   'edss_as_evaluated_by_clinician_05', 'edss_as_evaluated_by_clinician_06',
                                   'edss_as_evaluated_by_clinician_07', 'edss_as_evaluated_by_clinician_08',
                                   'edss_as_evaluated_by_clinician_09', 'edss_as_evaluated_by_clinician_10',
                                   'delta_edss_time0_01',
                                   'delta_edss_time0_02', 'delta_edss_time0_03',
                                   'delta_edss_time0_04', 'delta_edss_time0_05', 'delta_edss_time0_06',
                                   'delta_edss_time0_07', 'delta_edss_time0_08', 'delta_edss_time0_09'],
                       "object": ['sex', 'residence_classification', 'ethnicity', 'other_symptoms', 'centre',
                                  'multiple_sclerosis_type_01', 'multiple_sclerosis_type_02', 'mri_area_label_01',
                                  'mri_area_label_02', 'mri_area_label_03', 'mri_area_label_04', 'mri_area_label_05',
                                  'lesions_T1_01', 'lesions_T1_02', 'lesions_T1_gadolinium_01',
                                  'lesions_T1_gadolinium_02',
                                  'new_or_enlarged_lesions_T2_01', 'new_or_enlarged_lesions_T2_02',
                                  'new_or_enlarged_lesions_T2_03', 'new_or_enlarged_lesions_T2_04', 'lesions_T2_01',
                                  'lesions_T2_02', 'number_of_total_lesions_T2_01', 'number_of_total_lesions_T2_02',
                                  'altered_potential_01', 'altered_potential_02', 'altered_potential_03',
                                  'altered_potential_04', 'altered_potential_05', 'altered_potential_06',
                                  'altered_potential_07', 'altered_potential_08', 'potential_value_01',
                                  'potential_value_02',
                                  'potential_value_03', 'potential_value_04', 'potential_value_05',
                                  'potential_value_06',
                                  'potential_value_07', 'potential_value_08', 'location_01', 'location_02',
                                  'location_03',
                                  'location_04', 'location_05', 'location_06', 'location_07', 'location_08'
                                  ]
                       }
    col_value_types = df.columns.to_series().groupby(df.dtypes).groups

    col_value_types = {f"{key}": value for key, value in col_value_types.items()}

    cat_names = [*col_value_types["bool"], *col_value_types["object"]]
    cont_names = [*col_value_types["int64"], *col_value_types["float64"]]
    cont_names.remove("outcome_occurred")
    cont_names.remove("outcome_time")

    return cat_names, cont_names


def splits_strategy(df, valid_pct):
    splits = RandomSplitter(valid_pct=valid_pct)(range_of(df))
    return splits


def fastai_ccnames(df):

    # TODO add option to add outcome_time/outcome_occurred_depending on the preceding calculation

    col_value_types = df.columns.to_series().groupby(df.dtypes).groups

    col_value_types = {f"{key}": value for key, value in col_value_types.items()}
    cat_names = [*col_value_types["bool"], *col_value_types["object"]]
    cont_names = [*col_value_types["int64"], *col_value_types["float64"]]

    cont_names.remove("outcome_occurred")
    cont_names.remove("outcome_time")

    return cat_names, cont_names


def fastai_tab(df, cat_names, cont_names, y_names, splits):
    to = TabularPandas(df, procs=[Categorify, FillMissing, Normalize],
                       cat_names=cat_names,
                       cont_names=cont_names,
                       y_names=("outcome_occurred", "outcome_time"),
                       splits=splits)

    X_train, y_train = to.train.xs, to.train.ys.values.ravel()
    X_valid, y_valid = to.valid.xs, to.valid.ys.values.ravel()

    def y_to_struct_array(y, dtype):
        y = np.array(
            [(bool(outcome_occurred), outcome_time) for outcome_occurred, outcome_time in zip(y[::2], y[1::2])],
            dtype=dtype)
        return y

    struct_dtype = [('outcome_occurred', '?'), ('outcome_time', '<f8')]
    y_train = y_to_struct_array(y_train, dtype=struct_dtype)
    y_valid = y_to_struct_array(y_valid, dtype=struct_dtype)

    return X_train, y_train, X_valid, y_valid




def collapse_cols(df, feats_to_be_collapsed):

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

    for feat in feats_to_be_collapsed:
        df = collapse_ts_feature_cols(df, *feat)
    return df


def create_ts_features(df, features, drop_original=False):
    # df = df.copy()
    funcs = {"len": len,
             "max": np.max,
             "min": np.min,
             "sum": np.sum,
             "avg": np.average,
             "std": np.std}
    for feature in features:
        col_names = select_same_feature_col_names(df, feature)

        new_dfs = []
        for f_name, func in funcs.items():
            new_features_df = df_ts_func(df, col_names, func, f_name, new_feature_name=feature)
            new_dfs.append(new_features_df)

        df = pd.concat([df, *new_dfs], axis=1)

        if drop_original:
            df = df.drop(col_names, axis=1)

    return df


def select_same_feature_col_names(df, feature) -> list:
    return [col_name for col_name in df.columns.values.tolist() if col_name.startswith(feature)]





def df_one_hot_encode(original_dataframe, feature_to_encode, drop_org=False):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]]).astype(dtype=np.int64)
    res = pd.concat([original_dataframe, dummies], axis=1)
    if drop_org:
        res = res.drop(feature_to_encode, axis=1)
    return res


def df_ts_func(df, feature, func, func_name, new_feature_name):
    calculated_values = []
    for index, ts_row in df[feature].iterrows():
        ts_row = ts_row.dropna()
        if ts_row.empty:
            calculated_values.append(None)
        else:
            calculated_values.append(func(ts_row))
    out = {f"{new_feature_name}_{func_name}": calculated_values}
    return pd.DataFrame(out)


def df_normalize_col(df, feature):
    pass
