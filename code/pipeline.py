import math
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

from io_dfs import read_dfs, save_predictions, read_txt, filenames_in_folder
from merge_dfs import merge_dfs, merge_multiple
from preprocessing import preprocess, fastai_ccnames, fastai_tab, fastai_fill_split_xy, y_to_struct_array
from classification import init_classifiers, fit_models
from regressors import init_regressors
from evaluation import evaluate_c, evaluate_cumulative, plot_coef_, wrap_c_scorer
from survEstimators import init_surv_estimators, init_model, SurvTraceWrap, AvgEnsemble
from sklearn.model_selection import StratifiedKFold, GroupKFold, GroupShuffleSplit, ShuffleSplit
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from wandbsetup import setup_wandb, launch_sweep
import wandb
from sklearn import set_config

from load_survtrace import load_data
from survtrace.survtrace.evaluate_utils import Evaluator
from survtrace.survtrace.utils import set_random_seed
from survtrace.survtrace.model import SurvTraceSingle
from survtrace.survtrace.train_utils import Trainer
from survtrace.survtrace.config import STConfig


set_config(display="text")  # displays text representation of estimators


class IDPPPipeline:
    OUTPUT_DIR = "../out"
    num_iter = 100
    train_size = 0.8
    n_estimators = 100

    def __init__(self, dataset_dir, dataset_name, id_feature, seed):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.id_feature = id_feature
        self.seed = seed


        dataset_dirs = [f"../data/{dataset_name}_train", f"../data/{dataset_name}_test"]
        # dataset_dirs = [f"../data/datasetA_train", f"../data/datasetA_train_test",
        #                 f"../data/datasetB_train", f"../data/datasetB_train_test"]

        multiple_merge_dfs = []
        for data_dir, dataset_type in zip(dataset_dirs, ["train", "test", ]):
            dfs = read_dfs(data_dir)
            multiple_merge_dfs.append(merge_dfs(dfs, self.dataset_name, self.id_feature, dataset_type))

        self.patient_ids, self.merged_df = merge_multiple(multiple_merge_dfs, self.id_feature)
        self.train_ids, self.test_ids = self.patient_ids

        self.merged_df = preprocess(self.merged_df)

        self.X, self.y, _ = fastai_fill_split_xy(self.merged_df, self.seed)

        self.X_test, self.y_test = self.X.loc[self.test_ids], self.y.loc[self.test_ids]


        self.X, self.y = self.X.loc[self.train_ids], self.y.loc[self.train_ids]
        self.y_struct = y_to_struct_array(self.y, dtype=[('outcome_occurred', '?'), ('outcome_time', '<f8')])

        self.estimators = init_surv_estimators(self.seed, self.X, self.y, self.n_estimators)

        self.team_shortcut_t1 = "uwb_T1{}_{}"
        self.team_shortcut_t2 = "uwb_T2{}_{}"

        self.project = f"IDPP-CLEF-{dataset_name[-1]}{'_V3' if self.dataset_name == 'datasetA' else ''}"
        self.config = {"column_names": list(self.X.columns.values),
                       "X_shape": self.X.shape,
                       "num_iter": self.num_iter,
                       "train_size": self.train_size,
                       "seed": self.seed,
                       # "n_estimators": self.n_estimators
                       }

        self.notes = "(stat_vars[onehot])_(edss)_(delta_relapse_time0[funcs])_(evoked_potential[type][twosum])_final_avg"

    def run(self):
        best_accs, avg_acc, best_est = [], [], []
        for name, models in self.estimators.items():
            self.wandb_run = setup_wandb(project=self.project, config=self.config, name=name, notes=self.notes)
            print(f"Estimator: {name}")
            train_c_scores, val_c_scores, best_acc, best_estimator = self.run_n_times(models, num_iter=self.num_iter)
            best_accs.append(best_acc)
            avg_acc.append(np.average(val_c_scores))
            best_est.append(best_estimator)
            print(f"Train ({self.num_iter}iter) c-score: {np.average(train_c_scores)} ({np.std(train_c_scores)})")
            print(f"Val   ({self.num_iter}iter) c-score: {np.average(val_c_scores)} ({np.std(val_c_scores)})")

            if name == "CGBSA":
                plot_coef_(best_estimator, self.X)

            self.predict(best_estimator, self.X, self.y_struct, save=False)

            if name != "SurvTRACE":
                self.predict_cumulative(best_estimator, self.X, (self.y_struct, self.y_struct), save=False)

            # if name == "SurvTRACE_cumulative":
            #     self.predict_cumulative(best_estimator, self.X_test, save=True)

            self.wandb_run.finish()

        best_estimator_index = np.array(avg_acc).argmax()
        best_estimator = best_est[best_estimator_index]
        best_est_name = list(self.estimators.keys())[best_estimator_index]

        best_est_name = "AvgEnsemble"

        self.team_shortcut_t1 = self.team_shortcut_t1.format(self.dataset_name[-1].lower(), best_est_name)
        self.team_shortcut_t2 = self.team_shortcut_t2.format(self.dataset_name[-1].lower(), best_est_name)

        self.predict(best_estimator, self.X, self.y_struct, save=False)
        # self.predict(best_estimator, self.X_test, save=True)

        # if best_est_name != "SurvTRACE":
        #     self.predict_cumulative(best_estimator, self.X, (self.y_struct, self.y_struct), save=False)
        #     self.predict_cumulative(best_estimator, self.X_test, save=True)

        ensemble = AvgEnsemble(best_est)
        self.wandb_run = setup_wandb(project=self.project, config=self.config, name="EnsembleAvg", notes=self.notes)
        self.run_n_times(ensemble, 100)
        self.predict(ensemble, self.X, self.y_struct, save=False)
        self.predict(ensemble, self.X_test, save=True)
        # self.predict_cumulative(ensemble, self.X, (self.y_struct, self.y_struct), save=False)
        # self.predict_cumulative(ensemble, self.X_test, save=True)
        self.wandb_run.finish()

    def run_model(self, model, random_state):
        X, y_struct, y_df = self.X, self.y_struct, self.y
        avg_scores = {"train": [], "test": [], "model": []}
        # gss = GroupShuffleSplit(n_splits=10, train_size=self.train_size)
        # group_kfold = GroupKFold(n_splits=2)
        # groups = y_df["outcome_occurred"].to_numpy()

        ss = ShuffleSplit(n_splits=1, train_size=self.train_size, random_state=random_state)
        for i, (train_idx, test_idx) in enumerate(ss.split(X, y_struct)):
            X_train, y_train, X_valid, y_valid = X.iloc[train_idx], y_struct[train_idx], \
                                                 X.iloc[test_idx], y_struct[test_idx]

            if model.__class__.__name__ == "SurvTraceWrap":
                model = SurvTraceWrap(self.seed, X, y_df)
                model, train_c_score, test_c_score = model.fit(X, y_df, train_idx, test_idx)
            else:
                model.fit(X_train, y_train)
                train_c_score, _ = evaluate_c(model, X_train, y_train)
                test_c_score, _ = evaluate_c(model, X_valid, y_valid)
            avg_scores["train"].append(train_c_score)
            avg_scores["test"].append(test_c_score)
            avg_scores["model"].append(model)

        return avg_scores["train"], avg_scores["test"], avg_scores["model"]

    def run_n_times(self, model, num_iter=5):
        train_c_scores, val_c_scores, fitted_models = [], [], []
        for i in range(num_iter):
            error_flag = True
            while error_flag:
                try:
                    train_c_score, val_c_score, fitted_model = self.run_model(model, random_state=random.randint(0, 2**10))
                    train_c_scores += train_c_score
                    val_c_scores += val_c_score
                    fitted_models += fitted_model
                    error_flag = False

                except ValueError:
                    print("Error")
            # if i % 5 == 0 or (i+1) == num_iter:
            self.wandb_run.log({f"Num Iter": i+1,
                                f"Train C-Score Average": np.average(train_c_scores),
                                f"Val C-Score Average": np.average(val_c_scores),
                                f"Train C-Score": train_c_score[0],
                                f"Val C-Score": val_c_score[0],
                                f"Train C-Std": np.std(train_c_scores),
                                f"Val C-Std": np.std(val_c_scores)
                                })

        best_acc, best_estimator = (np.max(np.array(val_c_scores)), fitted_models[np.array(val_c_scores).argmax()])
        return np.array(train_c_scores), np.array(val_c_scores), best_acc, best_estimator

    def predict(self, best_model, X, y=None, save=False):
        c_score, predictions = evaluate_c(best_model, X, y)

        pred_output = {self.id_feature: X.index,
                       "predictions": predictions,
                       "run": self.team_shortcut_t1}

        print(best_model.__class__.__name__)
        print(f"Predictions C-Index Resized:", c_score)
        pred_df = pd.DataFrame(pred_output)

        if save:
            save_predictions(self.OUTPUT_DIR, self.team_shortcut_t1, pred_df)
        return pred_df

    def predict_cumulative(self, best_model, X, y=None, save=False):
        time_points = [2, 4, 6, 8, 10]
        auc_scores, predictions = evaluate_cumulative(best_model, X, y, time_points=time_points,
                                                      plot=True)

        pred_output = {self.id_feature: X.index,
                       "2years": predictions[:, 0],
                       "4years": predictions[:, 1],
                       "6years": predictions[:, 2],
                       "8years": predictions[:, 3],
                       "10years": predictions[:, 4],
                       "run": self.team_shortcut_t2}
        print("AUC Scores whole", auc_scores)
        pred_df = pd.DataFrame(pred_output)
        if save:
            save_predictions(self.OUTPUT_DIR, self.team_shortcut_t2, pred_df)

        return pred_df

    def param_sweep(self):
        sweep_id = launch_sweep(self.project, self.notes, self.config)
        wandb.agent(sweep_id, function=self.train_sweep_wrap)

    def train_sweep_wrap(self, ):
        with wandb.init():
            params = {**wandb.config}
            model = SurvTraceWrap(self.seed, self.X, self.y, wandb, **params)
            train_c_score, val_c_score, fitted_model = self.run_model(model, self.seed)
            wandb.log({f"Train C-Score": train_c_score[0],
                       f"Val C-Score": val_c_score[0],
                       })

    def plot_submission_results(self):
        file_dir = "../score/task1/"
        file_names = filenames_in_folder(file_dir)
        for file_name in file_names:
            if "T1b" not in file_name:
                continue
            submitted_predictions = read_txt(f"{file_dir}{file_name}")
            #
            # c_score = concordance_index_censored(self.y_test["outcome_occurred"].astype(bool),
            #                                      self.y_test["outcome_time"],
            #                                      submitted_predictions.iloc[:, 1])
            # print(file_name, c_score)

            # struct_dtype = [('outcome_occurred', '?'), ('outcome_time', '<f8')]
            # y_s = y_to_struct_array(self.y_test[["outcome_occurred", "outcome_time"]], struct_dtype)
            # auc_scores = cumulative_dynamic_auc(y_s, y_s, submitted_predictions.iloc[:, 1], [2, 4, 6, 8, 10])
            # print(file_name, auc_scores)
            submitted_predictions = submitted_predictions.to_numpy()[0]
            print(submitted_predictions)
            print(
                f"{submitted_predictions[0]} AUC: {submitted_predictions[2:17:3]} | Avg: {np.average(submitted_predictions[2:17:3])}\n{'=' * 80}")

        def plot_CIndex(file_names):
            score_data = [read_txt(f"{file_dir}{file_name}") for file_name in file_names if "T1b" in file_name]
            score_df = pd.concat(score_data)
            score_df[0] = ["-".join(name.removesuffix(".txt").split("_")[-2:]) if "minVal" in name else
                           name.removesuffix(".txt").split("_")[-1] for name in score_df[0]]
            score_df.columns = ["name", "lc", "c", "hc", "count"]
            score_df = score_df.sort_values(by="c", ascending=True)

            plt.style.use(["seaborn-v0_8-whitegrid"])
            import matplotlib.pyplot as mpl
            mpl.rcParams['font.size'] = 10

            fig, ax = plt.subplots()
            fig.subplots_adjust(left=0.28)

            for i, interval in enumerate(score_df[["lc", "hc"]].values):
                xerr = (interval[1] - interval[0]) / 2
                ax.errorbar(
                    score_df["c"].iloc[i], score_df["name"].iloc[i],  xerr=xerr, linestyle='', marker='_',
                    capsize=4, color="indigo"
                )
            ax.scatter(score_df["c"], score_df["name"],  color="black",)

            ax.set_xlabel('C-Index', fontweight='bold')
            ax.set_ylabel('Submitted run', fontweight='bold')
            ax.set_xlim((0, 1))
            # ax.set_xticks(np.arange(0, 1.01, step=0.05))
            ax.set_title('C-Index with Confidence intervals, Dataset B', fontsize=14, fontweight='bold')
            # ax.legend()
            # plt.grid()
            plt.savefig("../graphs/CIndexB.eps")
            plt.show()




    def run_ensemble(self):
        from sksurv.meta import EnsembleSelection, EnsembleSelectionRegressor, Stacking

        model = EnsembleSelectionRegressor([(name, est) for name, est in self.estimators.items()], scorer=wrap_c_scorer,
                                           n_jobs=10)

        self.wandb_run = setup_wandb(project=self.project, config=self.config, name="EnsembleSelection",
                                     notes=self.notes)
        print(f"Estimator: EnsembleSelection")
        train_c_scores, val_c_scores, best_acc, best_estimator = self.run_n_times(model, num_iter=self.num_iter)
        print(f"Train ({self.num_iter}iter) c-score: {np.average(train_c_scores)} ({np.std(train_c_scores)})")
        print(f"Val   ({self.num_iter}iter) c-score: {np.average(val_c_scores)} ({np.std(val_c_scores)})")

        self.wandb_run.log({f"Num Iter": self.num_iter,
                            f"Train C-Score Average": np.average(train_c_scores),
                            f"Val C-Score Average": np.average(val_c_scores),
                            f"Train C-Std": np.std(train_c_scores),
                            f"Val C-Std": np.std(val_c_scores)
                            })

        # best_estimator = est[np.array(acc).argmax()]

        self.predict(best_estimator)
        self.predict_cumulative(best_estimator, self.merged_df)

        self.wandb_run.finish()


def main():
    DEFAULT_RANDOM_SEED = 2021  # 2021  # random.randint(0, 2**10)

    def seed_basic(seed=DEFAULT_RANDOM_SEED):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)

    seed_basic(DEFAULT_RANDOM_SEED)

    DATASET = "datasetB"
    DATASET_DIR = f"../data/{DATASET}_train"
    ID_FEAT = "patient_id"

    pipeline = IDPPPipeline(DATASET_DIR, DATASET, ID_FEAT, DEFAULT_RANDOM_SEED)
    pipeline.plot_submission_results()
    # pipeline.run()
    # pipeline.param_sweep()

if __name__ == "__main__":
    main()
