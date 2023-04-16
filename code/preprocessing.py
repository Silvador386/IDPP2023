import pandas as pd
import numpy as np
from fastai.tabular.all import RandomSplitter, range_of, TabularPandas,\
    Categorify, FillMissing, Normalize, CategoryBlock

from time_window_functions import count_not_nan, mode_wrapper

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
    merged_df = merged_df.copy()

    features_to_encode = ["sex", "residence_classification", "ethnicity", 'ms_in_pediatric_age', "centre",
                          'spinal_cord_symptom', 'brainstem_symptom', 'eye_symptom', 'supratentorial_symptom',
                          "other_symptoms"
                          ]
    # for feature_cols in features_to_encode:
    #     merged_df = df_one_hot_encode(merged_df, feature_cols, drop_org=True)

    ts_features = ["age_at_onset", "edss_as_evaluated_by_clinician", "delta_edss_time0", "potential_value",
                   "delta_evoked_potential_time0", "delta_relapse_time0"]

    available_functions = {"mean": np.nanmean,
                           "std": np.nanstd,
                           "median": np.nanmedian,
                           "mode": mode_wrapper
                           }

    # time_windows = [(-730, -365), (-365, -183), (-182, 0), (-730, 0), (-365, 0)]
    time_windows = [(-1095, -730), (-730, -549), (-549, -365), (-365, -183), (-1095, 0), (-730, 0), (-549, 0), (-365, 0), (-183, 0)]

    merged_df = create_tw_features(merged_df, "edss_as_evaluated_by_clinician", "delta_edss_time0", available_functions,
                                   time_windows, drop_original=True)

    # time_windows = [(-1095, -730), (-730, -549), (-549, -365), (-365, -183),  (-182, 0)]
    merged_df = preprocess_evoked_potentials(merged_df, ['altered_potential', 'potential_value', 'location'],
                                 'delta_evoked_potential_time0', time_windows, drop_original=True)

    # merged_df = create_tw_features(merged_df, "potential_value", "delta_evoked_potential_time0",
    #                                {"sum": np.nansum}, time_windows, drop_original=True)


    # merged_df = create_tw_features(merged_df, "delta_relapse_time0", "delta_relapse_time0",  # Worsens score
    #                                {"enum": count_not_nan}, time_windows, drop_original=True)

    target_features = ['outcome_occurred', 'outcome_time']

    merged_df = merged_df.set_index("patient_id")

    features_to_leave = [*features_to_encode, *ts_features, *target_features,
                         "patient_id", "time_since_onset", "diagnostic_delay",
                         'multiple_sclerosis_type', 'delta_observation_time0',  # Might worsen the score
                         # "altered_potential"
                         ]

    unfinished_features = list(set(ALL_FEATURES).difference(features_to_leave))
    cols_to_drop = []
    for un_feat in unfinished_features:
        cols_to_drop += select_same_feature_col_names(merged_df, un_feat)
    merged_df = merged_df.drop(cols_to_drop, axis=1)

    merged_df = collapse_cols_as_occurrence_sum(merged_df, [("delta_relapse_time0", 6, None)])
    return merged_df


def splits_strategy(df, valid_pct, use_Kfold=True):
    splits = RandomSplitter(valid_pct=valid_pct)(range_of(df))

    return splits


def fastai_ccnames(df):

    # TODO add option to add outcome_time/outcome_occurred_depending on the preceding calculation

    col_value_types = df.columns.to_series().groupby(df.dtypes).groups

    col_value_types = {f"{key}": value for key, value in col_value_types.items()}
    cat_names = [*col_value_types["bool"], *col_value_types["object"], *col_value_types["int32"]]
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


def collapse_cols_as_occurrence_sum(df, feats_to_be_collapsed):

    def collapse_ts_feature_cols(df, feature, start_idx, end_idx=None):
        selected_cols = [col_name for col_name in df.columns.values.tolist() if col_name.startswith(feature) and col_name[-2:].isdigit()]
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


def select_same_feature_col_names(df, feature) -> list:
    return [col_name for col_name in df.columns.values.tolist() if col_name.startswith(feature)]


def df_one_hot_encode(original_dataframe, feature_to_encode, drop_org=False):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]]).astype(dtype=np.int64)
    res = pd.concat([original_dataframe, dummies], axis=1)
    if drop_org:
        res = res.drop(feature_to_encode, axis=1)
    return res


def create_tw_features(df, feature, time_feature, functions, time_windows, drop_original=False):
    feature_cols = select_same_feature_col_names(df, feature)
    time_cols = select_same_feature_col_names(df, time_feature)

    output_features = {}
    for (start_time, end_time) in time_windows:
        selected_values = select_time_window_values(df, feature_cols, time_cols, start_time, end_time)

        for func_name, func in functions.items():
            calculated_values = np.apply_along_axis(func, 1, selected_values)
            new_feature_name = f"{feature}_({start_time}_{end_time})_{func_name}"
            output_features[new_feature_name] = calculated_values

    new_df = pd.DataFrame(output_features)
    output_df = pd.concat([df, new_df], axis=1)

    if drop_original:
        output_df = output_df.drop([*feature_cols, *time_cols], axis=1)

    return output_df


def select_time_window_values(df, feature_cols, time_feature_cols, start_time=0, end_time=0):
    feature_matrix = df[feature_cols].to_numpy()
    time_values_matrix = df[time_feature_cols].to_numpy()

    time_mask = (start_time <= time_values_matrix) & (time_values_matrix <= end_time)

    selected_values = np.where(time_mask, feature_matrix, np.nan)

    return selected_values


def preprocess_evoked_potentials(df, feature_names, time_feature, time_windows, drop_original=False):
    features_cols = {feature: select_same_feature_col_names(df, feature) for feature in feature_names}

    time_cols = select_same_feature_col_names(df, time_feature)

    # altered potentials: ["Auditory", "Motor", "Somatosensory", "Visual"]
    # location: ["left", "lower left", "upper left", "right", "lower right", "upper right"]
    output_data = {}
    for (start_time, end_time) in time_windows:
        selected_data = {feat_name: select_time_window_values(df, feat_cols, time_cols, start_time, end_time) for feat_name, feat_cols in features_cols.items()}

        potential_value = selected_data["potential_value"]
        altered_potential = selected_data["altered_potential"]
        location = selected_data["location"]
        for altered_p_type in ["Auditory", "Motor", "Somatosensory", "Visual"]:
            mask = altered_potential == altered_p_type
            potential_value_masked = np.where(mask, potential_value, np.nan)
            calculated_values = np.apply_along_axis(np.nansum, 1, potential_value_masked)
            new_feature_name = f"altered_potential_({start_time}_{end_time})_{altered_p_type}"
            output_data[new_feature_name] = calculated_values

    new_df = pd.DataFrame(output_data)
    output_df = pd.concat([df, new_df], axis=1)

    if drop_original:
        output_df = output_df.drop([*[f_name for feature_cols in features_cols.values() for f_name in list(feature_cols)], *time_cols], axis=1)

    return output_df

