import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from read_dfs import read_dfs
from merge_dfs import merge_dfs
from preprocessing import preprocess, fastai_splits, fastai_tab
from classification import init_classifiers, fit_models
from regressors import init_regressors
from evaluation import clr_acc, evaluate_regressors_rmsle



class IDPPPipeline:
    def __init__(self, dataset_dir, dataset_name, id_feature):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.id_feature = id_feature
        self.dfs = None
        self.merged_df = None

    def run(self):
        self.dfs = read_dfs(self.dataset_dir)
        self.merged_df = merge_dfs(self.dfs, self.dataset_name, self.id_feature)
        self.merged_df = preprocess(self.merged_df)

        cat_names, cont_names, splits = fastai_splits(self.merged_df)
        X_train, y_train, X_valid, y_valid = fastai_tab(self.merged_df, cat_names, cont_names,
                                                        "outcome_occurred", splits)
        classifiers = init_classifiers()  # TODO turn of unwanted warnings.
        classifiers = fit_models(classifiers, X_train, y_train)
        classifiers_accuracies = clr_acc(classifiers, X_train, y_train, X_valid, y_valid)
        print(classifiers_accuracies)

        regressors = init_regressors()
        regressors = fit_models(regressors, X_train, y_train)

        evaluate_regressors_rmsle(regressors, X_train, y_train)
        evaluate_regressors_rmsle(regressors, X_valid, y_valid)





def main():
    DATASET = "datasetA"
    DATASET_DIR = f"../data/{DATASET}_train"
    ID_FEAT = "patient_id"

    pipeline = IDPPPipeline(DATASET_DIR, DATASET, ID_FEAT)
    pipeline.run()


if __name__ == "__main__":
    main()
