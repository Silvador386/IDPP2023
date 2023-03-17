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

DATASET_DIR = "../data/datasetA_train"
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
    dfs = {file_name.removesuffix(".csv"): pd.read_csv(os.path.join(dir_path, file_name)) for file_name in file_names if
           file_name.endswith("csv")}
    return dfs


dfs = read_dfs(DATASET_DIR)


def dfs_unique_patients(dfs):
    out = {}
    for name, df in dfs.items():
        out[name] = len(df[ID_FEAT].unique())
    return pd.DataFrame.from_dict(out, orient="index", columns=["Unique patients"])


unique_patients = dfs_unique_patients(dfs)
print(unique_patients)


def merge_dfs_complete(dfs):

    # def change_df(orig_df, id_feat, other_feature):
    #     orig_df = orig_df.sort_values(by=[id_feat, other_feature])
    #     unique_ids = orig_df[id_feat].unique()
    #     for

    def change_ts_column_to_list(orig_df, id_feat, other_feature):
        unique_ids = orig_df[id_feat].unique()
        grouped_ts_column_as_list = orig_df.groupby(id_feat)[other_feature].apply(lambda x: x.values).values.tolist()
        new_df = pd.DataFrame.from_dict({id_feat: unique_ids,
                                         other_feature: grouped_ts_column_as_list})
        new_df = new_df.set_index(id_feat)
        return new_df

    def change_ts_column_select_first(orig_df, id_feat, other_feature):
        unique_ids = orig_df[id_feat].unique()
        first_values = orig_df.groupby(id_feat)[other_feature].apply(lambda x: x.values[0]).values
        new_df = pd.DataFrame.from_dict({id_feat: unique_ids,
                                         other_feature: first_values})
        new_df = new_df.set_index(id_feat)
        return new_df

    def group_df_by_patient_id(orig_df, id_feat, time_series_feats, one_occurrence_feats, sort_by_this_feat):
        orig_df = orig_df.sort_values(by=[ID_FEAT, sort_by_this_feat])
        ts_dfs = [change_ts_column_to_list(orig_df, id_feat, ts_feat) for ts_feat in time_series_feats]
        oo_dfs = [change_ts_column_select_first(orig_df, id_feat, oo_feat) for oo_feat in one_occurrence_feats]

        out_df = pd.concat([*oo_dfs, *ts_dfs], axis=1)
        out_df.reset_index(names=id_feat, inplace=True)
        return out_df

    merged_df = pd.merge(dfs["datasetA_train-static-vars"], dfs["datasetA_train-outcomes"],
                         on="patient_id", how="outer")

    relapses_df = dfs["datasetA_train-relapses"]
    ts_feats = ["delta_relapse_time0"]
    oo_feats = ["centre"]
    sort_by_this_feat = "delta_relapse_time0"
    relapses_df = group_df_by_patient_id(relapses_df, ID_FEAT, ts_feats, oo_feats, sort_by_this_feat)

    ms_type_df = dfs["datasetA_train-ms-type"]
    ts_feats = ["multiple_sclerosis_type", "delta_observation_time0"]
    sort_by_this_feat = "delta_observation_time0"
    ms_type_df = group_df_by_patient_id(ms_type_df, ID_FEAT, ts_feats, oo_feats, sort_by_this_feat)

    #     mri_df = dfs["datasetA_train-mri.csv"]
    #     ts_feats = ["mri_area_label", "lesions_T1", "lesions_T1_gadolinium", "number_of_lesions_T1_gadolinium",
    #                 "new_or_enlarged_lesions_T2", "number_of_new_or_enlarged_lesions_T2", "lesions_T2", "number_of_total_lesions_T2", "delta_mri_time0"]
    #     oo_feats = ["centre"]
    #     mri_df = transpose_df_by_uniques(mri_df, ID_FEAT, ts_feats, oo_feats)

    #     evoked_p_df = dfs["datasetA_train-evoked-potentials.csv"]
    #     ts_feats = ["altered_potential", "potential_value", "location", "delta_evoked_potential_time0"]
    #     oo_feats = ["centre"]
    #     evoked_p_df = transpose_df_by_uniques(evoked_p_df, ID_FEAT, ts_feats, oo_feats)

    #     edss_df = dfs["datasetA_train-edss.csv"]
    #     ts_feats = ["edss_as_evaluated_by_clinician", "delta_edss_time0"]
    #     oo_feats = ["centre"]
    #     edss_df = transpose_df_by_uniques(edss_df, ID_FEAT, ts_feats, oo_feats)

    transposed_dfs = [relapses_df, ms_type_df]
    merged_df = pd.concat([merged_df, *transposed_dfs], axis=1)
    # print(relapses_df.columns.to_series().groupby(relapses_df.dtypes).groups)
    # print(merged_df.columns.to_series().groupby(merged_df.dtypes).groups)
    # merged_df = pd.merge(merged_df, relapses_df, on=ID_FEAT, how="outer")

    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()].copy()  # Removes duplicate patient_id, centre
    return merged_df


dfs = read_dfs(DATASET_DIR)
merge_dfs_complete(dfs)
