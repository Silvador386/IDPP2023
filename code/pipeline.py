import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from read_dfs import read_dfs
from merge_dfs import merge_dfs
from preprocessing import preprocess, fastai_ccnames, fastai_tab, fastai_ccnames_original, splits_strategy
from classification import init_classifiers, fit_models
from regressors import init_regressors
from evaluation import clr_acc, evaluate_regressors_rmsle, evaluate_estimators
from surestimators import init_surv_estimators

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn import set_config
set_config(display="text")  # displays text representation of estimators


class IDPPPipeline:
    def __init__(self, dataset_dir, dataset_name, id_feature):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.id_feature = id_feature
        self.dfs = None
        self.merged_df = None

        self.estimators = init_surv_estimators()

    def run(self):
        self.dfs = read_dfs(self.dataset_dir)
        self.merged_df = merge_dfs(self.dfs, self.dataset_name, self.id_feature)
        self.merged_df = preprocess(self.merged_df)

        num_iter = 3
        for name, models in self.estimators.items():
            print(f"Estimator: {name}")
            train_c_scores, val_c_scores, best_estimator = self.average_c_score(models, num_iter=num_iter)
            print(f"Train ({num_iter}iter) c-score: {np.average(train_c_scores)} ({np.std(train_c_scores)})")
            print(f"Val   ({num_iter}iter) c-score: {np.average(val_c_scores)} ({np.std(val_c_scores)})")

        self.predict_output_naive(best_estimator, self.merged_df)

    def run_model(self, model):
        splits = splits_strategy(self.merged_df, 0.2)
        cat_names, cont_names = fastai_ccnames_original(self.merged_df)
        X_train, y_train, X_valid, y_valid = fastai_tab(self.merged_df, cat_names, cont_names,
                                                        "outcome_occurred", splits)

        model.fit(X_train, y_train)

        train_c_score = evaluate_estimators(model, X_train, y_train, plot=False)
        test_c_score = evaluate_estimators(model, X_valid, y_valid, plot=False, print_coef=False)

        return train_c_score, test_c_score, model

    def average_c_score(self, model, num_iter=5):
        train_c_scores, val_c_scores, models = [], [], []
        for _ in range(num_iter):
            error_flag = True
            while error_flag:
                try:
                    train_c_score, val_c_score, model = self.run_model(model)
                    train_c_scores.append(train_c_score)
                    val_c_scores.append(val_c_score)
                    models.append(model)
                    error_flag = False

                except AssertionError:
                    print("Error")

        best_estimator = models[np.array(val_c_scores).argmax()]
        return np.array(train_c_scores), np.array(val_c_scores), best_estimator


    def predict_output_naive(self, best_model, df):
        splits = splits_strategy(df, valid_pct=0)
        cat_names, cont_names = fastai_ccnames_original(df)
        X_train, y_train, X_valid, y_valid = fastai_tab(df, cat_names, cont_names, "", splits)

        prediction_scores = best_model.predict(X_train)

        pred_df = pd.DataFrame([df[self.id_feature], prediction_scores])
        return pred_df


def main():
    DATASET = "datasetA"
    DATASET_DIR = f"../data/{DATASET}_train"
    ID_FEAT = "patient_id"

    pipeline = IDPPPipeline(DATASET_DIR, DATASET, ID_FEAT)
    pipeline.run()


if __name__ == "__main__":
    main()
