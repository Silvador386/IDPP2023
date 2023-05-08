import pandas as pd
import numpy as np
from fastai.tabular.all import RandomSplitter, range_of, TabularPandas, \
    Categorify, FillMissing, Normalize, CategoryBlock

from time_window_functions import count_not_nan, mode_wrapper, select_max_mri

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

FINISHED_FEATURES = ["sex", "residence_classification", "ethnicity", 'ms_in_pediatric_age', "centre",
                     'spinal_cord_symptom', 'brainstem_symptom', 'eye_symptom', 'supratentorial_symptom',
                     "other_symptoms",
                     "age_at_onset", "edss_as_evaluated_by_clinician", "delta_edss_time0", "potential_value",
                     "delta_evoked_potential_time0", "delta_relapse_time0",
                     'outcome_occurred', 'outcome_time'
                     "patient_id",
                     "time_since_onset", "diagnostic_delay",
                     'multiple_sclerosis_type', 'delta_observation_time0',  # Might worsen the score
                     "altered_potential",
                     'mri_area_label',
                     'lesions_T1', 'lesions_T1_gadolinium',
                     'number_of_lesions_T1_gadolinium', 'new_or_enlarged_lesions_T2',
                     'number_of_new_or_enlarged_lesions_T2', 'lesions_T2',
                     'number_of_total_lesions_T2', 'delta_mri_time0'
                     ]

feats_to_be_collapsed = [("new_or_enlarged_lesions_T2", 5, None),
                         ("number_of_new_or_enlarged_lesions_T2", 5, None),
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
    for feature_col in features_to_encode:
        merged_df = df_one_hot_encode(merged_df, feature_col, drop_org=True)

    # ts_features = ["age_at_onset", "edss_as_evaluated_by_clinician", "delta_edss_time0", "potential_value",
    #                "delta_evoked_potential_time0", "delta_relapse_time0"]

    available_functions = {"mean": np.nanmean,
                           "std": np.nanstd,
                           "median": np.nanmedian,
                           "mode": mode_wrapper
                           }

    # time_windows = [(-730, -365), (-365, -183), (-182, 0), (-730, 0), (-365, 0)]
    time_windows = [(-913, -730), (-730, -549), (-549, -365), (-365, -183),
                    (-913, 0), (-730, 0), (-549, 0), (-365, 0), (-183, 0)]

    merged_df = create_tw_features(merged_df, "edss_as_evaluated_by_clinician", "delta_edss_time0", available_functions,
                                   time_windows, drop_original=True)

    # time_windows = [(-1095, -730), (-730, -549), (-549, -365), (-365, -183),  (-182, 0)]
    merged_df = preprocess_evoked_potentials(merged_df, ['altered_potential', 'potential_value', 'location'],
                                             'delta_evoked_potential_time0', time_windows, drop_original=True)
    # # TODO interpolate over edss fillings
    merged_df = create_tw_features(merged_df, "delta_relapse_time0", "delta_relapse_time0",
                                   available_functions, time_windows, drop_original=True)

    # merged_df = preprocess_mri(merged_df,
    #                            ['mri_area_label',
    #                             'lesions_T1', 'lesions_T1_gadolinium',
    #                             'number_of_lesions_T1_gadolinium', 'new_or_enlarged_lesions_T2',
    #                             'number_of_new_or_enlarged_lesions_T2', 'lesions_T2', 'number_of_total_lesions_T2',
    #                             ],
    #                            "delta_mri_time0",
    #                            time_windows,
    #                            functions=available_functions,
    #                            drop_original=True)

    target_features = ['outcome_occurred', 'outcome_time']

    merged_df = merged_df.set_index("patient_id")

    features_to_leave = [*features_to_encode,  *target_features,
                         "patient_id", "edss_as_evaluated_by_clinician", "delta_edss_time0",
                         "delta_relapse_time0",
                         # "time_since_onset", "diagnostic_delay",
                         'altered_potential',
                         'potential_value', 'location', 'delta_evoked_potential_time0',
                         # 'multiple_sclerosis_type', #'delta_observation_time0',  # Might worsen the score
                         # "altered_potential",
                         # 'mri_area_label',
                         # 'lesions_T1', 'lesions_T1_gadolinium',
                         # 'number_of_lesions_T1_gadolinium', 'new_or_enlarged_lesions_T2',
                         # 'number_of_new_or_enlarged_lesions_T2', 'lesions_T2',
                         # 'number_of_total_lesions_T2', 'delta_mri_time0'
                         ]

    unfinished_features = list(set(ALL_FEATURES).difference(features_to_leave))
    cols_to_drop = []
    for un_feat in unfinished_features:
        cols_to_drop += select_same_feature_col_names(merged_df, un_feat)
    merged_df = merged_df.drop(cols_to_drop, axis=1)

    # merged_df = fill_missing_edss(merged_df, "edss_as_evaluated_by_clinician", list(available_functions.keys()))
    # merged_df = collapse_cols_as_occurrence_sum(merged_df, feats_to_be_collapsed)  # [("delta_relapse_time0", 6, None)]
    return merged_df


def fastai_fill_split_xy(df, seed):
    splits = RandomSplitter(valid_pct=0.0, seed=seed)(range_of(df))
    cat_names, cont_names = fastai_ccnames(df)

    X, y, y_struct, _, __, ___ = fastai_tab(df, cat_names, cont_names, splits)
    return X, y, y_struct


def fastai_ccnames(df):
    # TODO add option to add outcome_time/outcome_occurred_depending on the preceding calculation

    col_value_types = df.columns.to_series().groupby(df.dtypes).groups

    col_value_types = {f"{key}": value for key, value in col_value_types.items()}

    cat_names, cont_names = [], []
    for type_name in col_value_types.keys():
        if type_name in ["int8", "int32",  "bool", "object"]:
            cat_names += col_value_types[type_name].to_list()
        else:
            cont_names += col_value_types[type_name].to_list()

    if "outcome_occurred" in cont_names:
        cont_names.remove("outcome_occurred")
    if "outcome_time" in cont_names:
        cont_names.remove("outcome_time")

    return cat_names, cont_names


def fastai_tab(df, cat_names, cont_names, splits):
    to = TabularPandas(df, procs=[Categorify, FillMissing, Normalize],
                       cat_names=cat_names,
                       cont_names=cont_names,
                       y_names=["outcome_occurred", "outcome_time"],
                       splits=splits)

    X_train, y_train_df = to.train.xs, to.train.ys
    X_valid, y_valid_df = to.valid.xs, to.valid.ys

    struct_dtype = [('outcome_occurred', '?'), ('outcome_time', '<f8')]
    y_train_struct = y_to_struct_array(y_train_df, dtype=struct_dtype)
    y_valid_struct = y_to_struct_array(y_valid_df, dtype=struct_dtype)

    return X_train, y_train_df, y_train_struct, X_valid, y_valid_df, y_valid_struct


def y_to_struct_array(y, dtype):
    y = np.array([(bool(outcome_occurred), outcome_time) for outcome_occurred, outcome_time in zip(y.iloc[:, 0], y.iloc[:, 1])],dtype=dtype)
    return y


def fill_missing_edss(df, feature, specific_names, drop_na_all=True):

    def func(row):
        mean_value = np.nanmean(row)
        flag_first_value = True
        for i, value in enumerate(row):
            if np.isnan(value):
                if flag_first_value:
                    row[i] = 0
                else:
                    row[i] = mean_value
            else:
                flag_first_value = False
        return row

    feature_cols = select_same_feature_col_names(df, feature)
    if drop_na_all:
        df = df[~df[feature_cols].isna().all(axis=1)].copy()
    for specific_name in specific_names:
        func_win_cols = [feat for feat in feature_cols if specific_name in feat]
        fill_value = 0  #df[func_win_cols].mean(axis=1, skipna=True)

        # func(df[func_win_cols])

        df[func_win_cols] = df[func_win_cols].apply(func, axis=1)


        # df[func_win_cols] = df[func_win_cols].T.fillna(fill_value).T

    # if drop_na_all:
    #     df = df[~df[feature_cols].isna().any(axis=1)]  # Drop rows where at least 1 value was nan
    return df


def collapse_cols_as_occurrence_sum(df, feats_to_be_collapsed):
    def collapse_ts_feature_cols(df, feature, start_idx, end_idx=None):
        selected_cols = [col_name for col_name in df.columns.values.tolist() if
                         col_name.startswith(feature) and col_name[-2:].isdigit()]
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


def select_same_feature_col_names_specific(df, feature) -> list:
    return [col_name for col_name in df.columns.values.tolist()
            if col_name.startswith(feature) and col_name[len(feature)+1:len(feature)+3].isdigit()]


def df_one_hot_encode(original_dataframe, feature_to_encode, drop_org=False):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]]).astype(dtype=np.int32)
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
        selected_data = {feat_name: select_time_window_values(df, feat_cols, time_cols, start_time, end_time) for
                         feat_name, feat_cols in features_cols.items()}

        potential_value = selected_data["potential_value"].astype(float)
        # potential_value = np.where(potential_value.astype(str) == "True", potential_value, True).astype(int)


        # altered_potential = selected_data["altered_potential"]
        # location = selected_data["location"]
        # for loc_type in ["Auditory", "Motor", "Somatosensory", "Visual"]:
        #     mask = (altered_potential == loc_type)
        #     potential_value_masked = np.where(mask, potential_value, np.nan)

        calculated_values = np.apply_along_axis(np.nansum, 1, potential_value)
        new_feature_name = f"altered_potential_({start_time}_{end_time})_Psum"
        output_data[new_feature_name] = calculated_values
        calculated_values = np.apply_along_axis(count_not_nan, 1, potential_value)
        new_feature_name = f"altered_potential_({start_time}_{end_time})_NNaNsum"
        output_data[new_feature_name] = calculated_values

    new_df = pd.DataFrame(output_data)
    output_df = pd.concat([df, new_df], axis=1)

    if drop_original:
        output_df = output_df.drop(
            [*[f_name for feature_cols in features_cols.values() for f_name in list(feature_cols)], *time_cols], axis=1)

    return output_df


"""
+ mri_area_label:
	The area on which the MRI has been performed. Possible values: {Brain Stem, Cervical Spinal Cord, Spinal Cord, Thoracic Spinal Cord}.

+ lesions_T1:
	Boolean variable that states whether the MRI observes some lesions in T1 (True) or not (False).

+ lesions_T1_gadolinium:
	Boolean variable that states whether there are some Gadolinium-enhancing lesions (True) or not (False).

+ number_of_lesions_T1_gadolinium:
	The number of Gadolinium-enhancing lesions.

+ new_or_enlarged_lesions_T2:
	Boolean variable that states whether there are new or enlarged lesions in T2 since last MRI (True) or not (False).

+ number_of_new_or_enlarged_lesions_T2:
	Number of new or enlarged lesions in T2 since last MRI.

+ lesions_T2:
	Boolean variable that states whether the MRI observes some lesions in T2 (True) or not (False).

+ number_of_total_lesions_T2:
	The number of total lesions in T2. When not absent, possible values are: {0, 1-2, >=3, >=9}.

+ delta_mri_time0:
	The date on which the MRI has been performed, expressed in days as relative delta with respect to Time 0 (mri_date - time0).
"""


def preprocess_mri(df, feature_names, time_feature, time_windows, functions, drop_original=False):
    features_cols = {feature: select_same_feature_col_names_specific(df, feature) for feature in feature_names}

    time_cols = select_same_feature_col_names(df, time_feature)

    output_data = {}
    for (start_time, end_time) in time_windows:
        selected_data = {feat_name: select_time_window_values(df, feat_cols, time_cols, start_time, end_time) for
                         feat_name, feat_cols in features_cols.items()}

        mri_area = selected_data["mri_area_label"]

        vals = {"0": 1, "1-2": 2, ">=3": 3, ">=9": 4, "nan": 0}
        total_num_cat = selected_data["number_of_total_lesions_T2"].astype(str)
        u, inv = np.unique(total_num_cat, return_inverse=True)
        total_num_cat = np.array([vals[x] for x in u])[inv].reshape(total_num_cat.shape)

        calculated_values = np.apply_along_axis(np.nanmax, 1, total_num_cat)
        new_feature_name = f"mri_({start_time}_{end_time})_totalT2"
        output_data[new_feature_name] = calculated_values

        lesions_T1 = selected_data["lesions_T1"]
        num_gan_T1 = selected_data["number_of_lesions_T1_gadolinium"]

        lesions_T2 = selected_data["lesions_T2"]
        num_new_T2 = selected_data["number_of_new_or_enlarged_lesions_T2"]

        # calculated_values = np.apply_along_axis(np.nansum, 1, num_gan_T1)
        # new_feature_name = f"mri_T1_gadolinium_({start_time}_{end_time})_sum"
        # output_data[new_feature_name] = calculated_values
        # calculated_values = np.apply_along_axis(count_not_nan, 1, num_gan_T1)
        # new_feature_name = f"mri_T1_gadolinium_({start_time}_{end_time})_nan"
        # output_data[new_feature_name] = calculated_values

        # mri-label: ["Brain Stem", "Cervical Spinal Cord", "Spinal Cord", "Thoracic Spinal Cord"]
        # for mri_label in ["Brain Stem", "Cervical Spinal Cord", "Spinal Cord", "Thoracic Spinal Cord"]:
        #     mask = (mri_area == mri_label)
        #
        #     for region in ["number_of_lesions_T1_gadolinium", "number_of_new_or_enlarged_lesions_T2"]:
        #         region_data = selected_data[region]
        #         region_masked = np.where(mask, region_data, np.nan)
        #         calculated_values = np.apply_along_axis(np.nansum, 1, region_masked)
        #         new_feature_name = f"mri_({start_time}_{end_time})_{mri_label}"
        #         output_data[new_feature_name] = calculated_values

            # for f_name, func in functions.items():
            #     calculated_values = np.apply_along_axis(func, 1, calculated_values)
            #     new_feature_name = f"mri_{mri_label}_({start_time}_{end_time})_{f_name}"
            #     output_data[new_feature_name] = calculated_values


            # for region in ["number_of_lesions_T1_gadolinium",
            #                "number_of_new_or_enlarged_lesions_T2"]:
            #     region_data = selected_data[region]
            #     region_masked = np.where(mask, region_data, np.nan)
            #     calculated_values = np.apply_along_axis(np.nansum, 1, region_masked)
            #     new_feature_name = f"mri_({start_time}_{end_time})_{mri_label}_{region}"
            #     output_data[new_feature_name] = calculated_values


    new_df = pd.DataFrame(output_data).fillna(0)
    output_df = pd.concat([df, new_df], axis=1)

    if drop_original:
        output_df = output_df.drop(
            [*[f_name for feature_cols in features_cols.values() for f_name in list(feature_cols)], *time_cols], axis=1)

    return output_df
