import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from read_dfs import read_dfs
from merge_dfs import merge_dfs
from preprocessing import preprocess, fastai_ccnames, fastai_tab, splits_strategy
from classification import init_classifiers, fit_models
from regressors import init_regressors
from evaluation import clr_acc, evaluate_regressors_rmsle, evaluate_estimators
from surestimators import init_surv_estimators
from sklearn.model_selection import StratifiedKFold, GroupKFold, GroupShuffleSplit
from sksurv.metrics import concordance_index_censored
from wandbsetup import setup_wandb
from sklearn import set_config

set_config(display="text")  # displays text representation of estimators


class IDPPPipeline:
    TEAM_SHORTCUT = "uwb_T1a_surfRF"

    def __init__(self, dataset_dir, dataset_name, id_feature):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.id_feature = id_feature

        self.dfs = read_dfs(self.dataset_dir)
        self.merged_df = merge_dfs(self.dfs, self.dataset_name, self.id_feature)
        self.merged_df = preprocess(self.merged_df)

        self.estimators = init_surv_estimators()

        config = {"column_names": self.merged_df.columns.values}
        self.wandb_run = setup_wandb(config)

    def run(self):

        num_iter = 3
        for name, models in self.estimators.items():
            print(f"Estimator: {name}")
            train_c_scores, val_c_scores, best_estimator = self.average_c_score(models, num_iter=num_iter)
            print(f"Train ({num_iter}iter) c-score: {np.average(train_c_scores)} ({np.std(train_c_scores)})")
            print(f"Val   ({num_iter}iter) c-score: {np.average(val_c_scores)} ({np.std(val_c_scores)})")

        self.predict_output_naive(best_estimator, self.merged_df)
        self.predict_cumulative(best_estimator, self.merged_df)

    def run_model(self, model):
        splits = splits_strategy(self.merged_df, 0.0)
        cat_names, cont_names = fastai_ccnames(self.merged_df)

        X_train0, y_train0, X_valid, y_valid = fastai_tab(self.merged_df, cat_names, cont_names,
                                                        "outcome_occurred", splits=splits)

        avg_scores ={"train": [], "test": []}
        gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=43)
        for i, (train_idx, test_idx) in enumerate(gss.split(X_train0, y_train0, groups=self.merged_df[["outcome_occurred"]])):

            X_train, y_train, X_valid, y_valid = X_train0.iloc[train_idx], y_train0[train_idx],\
                                                 X_train0.iloc[test_idx], y_train0[test_idx]


            model.fit(X_train, y_train)

            train_c_score = evaluate_estimators(model, X_train, y_train, plot=False)
            test_c_score = evaluate_estimators(model, X_valid, y_valid, plot=False, print_coef=False)
            avg_scores["train"].append(train_c_score)
            avg_scores["test"].append(test_c_score)

        return np.average(np.array(avg_scores["train"])), np.average(np.array(avg_scores["test"])), model

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
        cat_names, cont_names = fastai_ccnames(df)
        X, y, _, _ = fastai_tab(df, cat_names, cont_names, "", splits)

        prediction_scores = best_model.predict(X)  # TODO Find out how to scale scores to fit in <0, 1 interval
        prediction_scores = resize_prediction_score_vec(prediction_scores)

        pred_output = {self.id_feature: df.index,
                       "predictions": prediction_scores,
                       "run": self.TEAM_SHORTCUT}
        y_occ = [val[0] for val in y]
        y_time = [val[1] for val in y]
        print("Predictions C-Index", concordance_index_censored(y_occ, y_time, prediction_scores))
        pred_df = pd.DataFrame(pred_output)
        return pred_df

    def predict_cumulative(self, best_model, df):
        splits = splits_strategy(df, valid_pct=0)
        cat_names, cont_names = fastai_ccnames(df)
        X, y, _, _ = fastai_tab(df, cat_names, cont_names, "", splits)
        pred_surv = best_model.predict_cumulative_hazard_function(X)

        time_points = [2, 4, 6, 8, 10]
        predictions = []
        for i, surv_func in enumerate(pred_surv):
            predictions.append(surv_func(time_points))


            plt.step(time_points, surv_func(time_points), where="post",
                     label="Sample %d" % (i + 1))
        plt.title("CoxPHSurvival Step functions")
        plt.ylabel("est. probability of survival $\hat{S}(t)$")
        plt.xlabel("time $t$")
        plt.ylim([0, 1])
        plt.legend(loc="best")
        plt.show()

        predictions = np.array(predictions)

        predictions = resize_prediction_score_naive(predictions)

        pred_output = {self.id_feature: df.index,
                       "2years": predictions[:, 0],
                       "4years": predictions[:, 1],
                       "6years": predictions[:, 2],
                       "8years": predictions[:, 3],
                       "10years": predictions[:, 4],
                       "run": self.TEAM_SHORTCUT}

        pred_df = pd.DataFrame(pred_output)
        return pred_df


def resize_prediction_score_naive(prediction_scores):
    min = prediction_scores.min()
    max = prediction_scores.max()
    return (prediction_scores - min)/(max-min)


def resize_prediction_score_vec(prediction_scores):
    num_preds = len(prediction_scores)
    preds = [sum(value > prediction_scores) / num_preds for value in prediction_scores]
    return np.array(preds)

def main():
    DATASET = "datasetA"
    DATASET_DIR = f"../data/{DATASET}_train"
    ID_FEAT = "patient_id"

    pipeline = IDPPPipeline(DATASET_DIR, DATASET, ID_FEAT)
    pipeline.run()


if __name__ == "__main__":
    main()
