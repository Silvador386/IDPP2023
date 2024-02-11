import pandas as pd
import numpy as np

from utils.ioutils import load_dfs_from_files_in_dir
from merge_dfs import merge_dfs
from preprocessing import preprocess, fastai_preproccess_dataset, y_to_struct_array


class IDDPDataset:
    TARGET_NAMES = ['outcome_occurred', 'outcome_time']
    TARGET_STRUCT_PATTERN = [('outcome_occurred', '?'), ('outcome_time', '<f8')]


    def __init__(
            self,
            dataset_name: str,
            id_feature_name: str,
            seed: int
    ):

        self.dataset_name = dataset_name
        self.id_feature_name = id_feature_name
        self.seed = seed

        self.X, self.y, self.y_struct = None, None, None
        self.X_test, self.y_test, self.y_test_struct = None, None, None

    def load_data(self, train_dir_path: str, test_dir_path: str = None):

        train_dataset_sections = load_dfs_from_files_in_dir(train_dir_path)
        train_df = merge_dfs(train_dataset_sections, self.dataset_name, self.id_feature_name, dataset_type="train")
        train_ids = train_df[self.id_feature_name]
        data_df = train_df

        if test_dir_path:
            test_dataset_sections = load_dfs_from_files_in_dir(test_dir_path)
            test_df = merge_dfs(test_dataset_sections, self.dataset_name, self.id_feature_name, dataset_type="test")
            test_ids = test_df[self.id_feature_name]
            data_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

        data_df = preprocess(data_df)

        data_X, data_y = fastai_preproccess_dataset(data_df, target_names=self.TARGET_NAMES, seed=self.seed)

        self.X, self.y = data_X.loc[train_ids], data_y.loc[train_ids]
        self.y_struct = y_to_struct_array(self.y, dtype=self.TARGET_STRUCT_PATTERN)

        if test_dir_path:
            self.X_test, self.y_test = data_X.loc[test_ids], data_y.loc[test_ids]
            self.y_test_struct = y_to_struct_array(self.y_test, dtype=self.TARGET_STRUCT_PATTERN)

    def get_train_data(self) -> tuple[pd.DataFrame, pd.DataFrame, np.array]:
        return self.X, self.y, self.y_struct

    def has_test_data(self) -> bool:
        return self.X_test is not None and self.y_test_struct is not None

    def get_test_data(self) -> tuple[pd.DataFrame, pd.DataFrame, np.array]:
        return self.X_test, self.y_test, self.y_test_struct

    def get_feature_names(self) -> list:
        if self.X is not None:
            return self.X.columns.to_list()
        else:
            return []

    def get_dataset_metadata(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "id_feature_name": self.id_feature_name,
            "seed": self.seed,
            "column_names": self.get_feature_names(),
            "X_shape": self.X.shape,
            "y_shape": self.y.shape
        }





