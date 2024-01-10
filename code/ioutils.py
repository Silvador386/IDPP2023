import os
import os.path as osp
import pandas as pd


def filenames_in_folder(dir_path):
    file_names = []
    for _, __, f_names in os.walk(dir_path):
        for file_name in f_names:
            file_names.append(file_name)
        break
    return file_names


def load_df_from_file(file_path):
    return pd.read_csv(file_path, sep=" ", header=None)


def load_dfs_from_files_in_dir(dir_path):
    file_names = filenames_in_folder(dir_path)
    dfs = {file_name.removesuffix(".csv"): pd.read_csv(os.path.join(dir_path, file_name)) for file_name in file_names if
           file_name.endswith("csv")}
    return dfs


def save_predictions(dir_path, file_name, predictions):
    file_path = osp.join(dir_path, file_name)
    predictions.to_csv(file_path, header=None, index=None, sep=' ', mode='w+')
