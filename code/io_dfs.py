import os
import pandas as pd


def filenames_in_folder(dir_path):
    file_names = []
    for _, __, f_names in os.walk(dir_path):
        for file_name in f_names:
            file_names.append(file_name)
        break
    return file_names

def read_txt(file_path):
    return pd.read_csv(file_path, sep=" ", header=None)


def read_dfs(dir_path):
    file_names = filenames_in_folder(dir_path)
    dfs = {file_name.removesuffix(".csv"): pd.read_csv(os.path.join(dir_path, file_name)) for file_name in file_names if
           file_name.endswith("csv")}
    return dfs


def save_predictions(dir_path, file_name, predictions):
    file_path = dir_path + "/" + file_name + ".txt"
    predictions.to_csv(file_path, header=None, index=None, sep=' ', mode='w+')
