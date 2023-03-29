import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
import lightgbm

# machine learning
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import r_regression
from sklearn import preprocessing

plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
)


DATASET = "datasetA"
DATASET_DIR = f"../data/{DATASET}_train"
ID_FEAT = "patient_id"

pd.set_option('display.max_rows', None)


def filenames_in_folder(dir_path):
    file_names = []
    for _, __, f_names in os.walk(dir_path):
        for file_name in f_names:
            file_names.append(file_name)
        break
    return file_names


def read_dfs(dir_path):
    file_names = filenames_in_folder(dir_path)
    dfs = {file_name.removesuffix(".csv"): pd.read_csv(os.path.join(dir_path, file_name)) for file_name in file_names if file_name.endswith("csv")}
    return dfs


def dfs_unique_patients(dfs):
    out = {}
    for name, df in dfs.items():
        out[name] = len(df[ID_FEAT].unique())
    return pd.DataFrame.from_dict(out, orient="index", columns=["Unique patients"])


def merge_dfs_complete(dfs, dataset):

    def group_ts_column_to_list(orig_df, id_feat, other_feature):
        unique_ids = orig_df[id_feat].unique()
        grouped_ts_column_as_list = orig_df.groupby(id_feat)[other_feature].apply(lambda x: x.values).values.tolist()
        new_df = pd.DataFrame.from_dict({id_feat: unique_ids,
                                         other_feature: grouped_ts_column_as_list})
        new_df = new_df.set_index(id_feat)
        return new_df

    def group_ts_column_select_first(orig_df, id_feat, other_feature):
        unique_ids = orig_df[id_feat].unique()
        first_values = orig_df.groupby(id_feat)[other_feature].apply(lambda x: x.values[0]).values

        new_df = pd.DataFrame.from_dict({id_feat: unique_ids,
                                         other_feature: first_values})
        new_df = new_df.set_index(id_feat)
        return new_df

    def group_ts_df_by_id(orig_df, id_feat, time_series_feats, one_occurrence_feats, sort_by_this_feat):
        orig_df = orig_df.sort_values(by=[ID_FEAT, sort_by_this_feat])
        ts_dfs = [group_ts_column_to_list(orig_df, id_feat, ts_feat) for ts_feat in time_series_feats]
        oo_dfs = [group_ts_column_select_first(orig_df, id_feat, oo_feat) for oo_feat in one_occurrence_feats]

        out_df = pd.concat([*oo_dfs, *ts_dfs], axis=1)
        out_df.reset_index(names=id_feat, inplace=True)
        return out_df

    merged_df = pd.merge(dfs[f"{dataset}_train-static-vars"], dfs[f"{dataset}_train-outcomes"],
                         on="patient_id", how="outer")

    relapses_df = dfs[f"{dataset}_train-relapses"]
    ts_feats = ["delta_relapse_time0"]
    oo_feats = ["centre"]
    sort_by_this_feat = "delta_relapse_time0"
    relapses_df = group_ts_df_by_id(relapses_df, ID_FEAT, ts_feats, oo_feats, sort_by_this_feat)

    ms_type_df = dfs[f"{dataset}_train-ms-type"]
    ts_feats = ["multiple_sclerosis_type", "delta_observation_time0"]
    oo_feats = ["centre"]
    sort_by_this_feat = "delta_observation_time0"
    ms_type_df = group_ts_df_by_id(ms_type_df, ID_FEAT, ts_feats, oo_feats, sort_by_this_feat)

    mri_df = dfs[f"{dataset}_train-mri"]
    ts_feats = ["mri_area_label", "lesions_T1", "lesions_T1_gadolinium", "number_of_lesions_T1_gadolinium",
                "new_or_enlarged_lesions_T2", "number_of_new_or_enlarged_lesions_T2", "lesions_T2", "number_of_total_lesions_T2", "delta_mri_time0"]
    oo_feats = ["centre"]
    sort_by_this_feat = "delta_mri_time0"
    mri_df = group_ts_df_by_id(mri_df, ID_FEAT, ts_feats, oo_feats, sort_by_this_feat)

    evoked_p_df = dfs[f"{dataset}_train-evoked-potentials"]
    ts_feats = ["altered_potential", "potential_value", "location", "delta_evoked_potential_time0"]
    oo_feats = ["centre"]
    sort_by_this_feat = "delta_evoked_potential_time0"
    evoked_p_df = group_ts_df_by_id(evoked_p_df, ID_FEAT, ts_feats, oo_feats, sort_by_this_feat)


    edss_df = dfs[f"{dataset}_train-edss"]
    ts_feats = ["edss_as_evaluated_by_clinician", "delta_edss_time0"]
    oo_feats = ["centre"]
    sort_by_this_feat = "delta_edss_time0"
    edss_df = group_ts_df_by_id(edss_df, ID_FEAT, ts_feats, oo_feats, sort_by_this_feat)

    grouped_dfs = [edss_df, relapses_df, ms_type_df, evoked_p_df, mri_df]
    for df in grouped_dfs:
        merged_df = pd.merge(merged_df, df, on=[ID_FEAT, "centre"], how="outer")

    return merged_df


dfs = read_dfs(DATASET_DIR)
merged_df = merge_dfs_complete(dfs, DATASET)


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


merged_df = merged_df.replace({np.nan: None})
df_one_hot_encode(merged_df, "sex")

merged_df = preprocess(merged_df)
