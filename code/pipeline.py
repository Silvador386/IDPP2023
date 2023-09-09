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
        plt.style.use(["seaborn-v0_8-whitegrid"])
        import matplotlib.pyplot as mpl
        mpl.rcParams['font.size'] = 10


        task1_file_dir = "../score/task1/"
        task2_file_dir = "../score/task2/"
        task1_file_names = filenames_in_folder(task1_file_dir)

        dataset_type = self.dataset_name[-1]
        type_name = "Val"
        save_name = f"{type_name}C_{dataset_type}"

        def get_task1_scores_from_txt(file_dir, file_names):
            score_data = [read_txt(f"{file_dir}{file_name}") for file_name in file_names if f"T1{self.dataset_name[-1].lower()}" in file_name]
            score_df = pd.concat(score_data)
            score_df[0] = ["-".join(name.removesuffix(".txt").split("_")[-2:]) if "minVal" in name else
                           name.removesuffix(".txt").split("_")[-1] for name in score_df[0]]
            score_df.columns = ["name", "lc", "c", "hc", "count"]
            score_df = score_df.sort_values(by="c", ascending=True)

            name_map = {'SurvTRACE-minVal': "SurvTRACE - MinVal", 'survRF': "Random Forest", 'CGBSA': "CGBSA", 'survRFmri': "Random Forest MRI", 'survGB': "Gradient Boosting",
            "AvgEnsemble": 'Ensemble Avg.', 'AvgEnsemble-minVal': 'Ensemble Avg. - MinVal', 'survGB-minVal': 'Gradient Boosting - MinVal', 'SurvTRACE': "SurvTRACE"}
            score_df["name"].replace(name_map, inplace=True)
            return score_df

        def get_task2_scores(file_dir):
            file_names = filenames_in_folder(file_dir)
            score_data = [read_txt(f"{file_dir}{file_name}") for file_name in file_names if
                          f"T2{self.dataset_name[-1].lower()}" in file_name]
            score_df = pd.concat(score_data)
            score_df[0] = ["-".join(name.removesuffix(".txt").split("_")[-2:]) if "minVal" in name else
                           name.removesuffix(".txt").split("_")[-1] for name in score_df[0]]

            auroc_cols = [f"{'L' if i % 3 == 0 else '' if i % 3 == 1 else 'H'}AUROC (0-{(i // 3) * 2 + 2})" for i in
                          range(15)]
            oe_cols = [f"{'L' if i % 3 == 0 else '' if i % 3 == 1 else 'H'}O/E (0-{(i // 3) * 2 + 2})" for i in range(15)]
            col_names = ["name", *auroc_cols, *oe_cols, "count"]
            score_df.columns = col_names
            score_df = score_df.iloc[score_df.filter(regex=f"^{'AUROC'}").mean(axis=1).argsort()]  # sort by avg auroc
            score_df = score_df.iloc[::-1]

            name_map = {'SurvTRACE-minVal': "SurvTRACE - MinVal", 'survRF': "Random Forest", 'CGBSA': "CGBSA",
                        'survRFmri': "Random Forest MRI", 'survGB': "Gradient Boosting",
                        "AvgEnsemble": 'Ensemble Avg.', 'AvgEnsemble-minVal': 'Ensemble Avg. - MinVal',
                        'survGB-minVal': 'Gradient Boosting - MinVal', 'SurvTRACE': "SurvTRACE"}
            score_df["name"].replace(name_map, inplace=True)

            return score_df

        def make_spider(df, yticks, save_loc, ytick_bold: int = None, legend=True):
            import matplotlib.colors as mcolors

            colors = [*mcolors.TABLEAU_COLORS]
            categories = df.columns.values
            N = len(categories)

            angles = [n / float(N) * 2 * np.pi for n in range(N)]

            plt.rc('figure', figsize=(10, 8))
            ax = plt.subplot(1, 1, 1, polar=True)

            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            plt.xticks(angles, categories, color='black', size=18, weight='bold')
            ax.tick_params(axis='x', rotation=5.5)

            ax.set_rlabel_position(0)
            plt.yticks(yticks, color="black", size=18)
            y_lim_low_val = yticks[0]
            plt.ylim(y_lim_low_val, yticks[-1] + (yticks[-1] - yticks[-2]))

            if ytick_bold:
                labels = ax.get_yticklabels()
                labels[ytick_bold].set_fontweight('bold')
                labels[ytick_bold].set_fontsize(20)

                gridlines = ax.yaxis.get_gridlines()
                gridlines[ytick_bold].set_color("k")
                gridlines[ytick_bold].set_linewidth(3)

            angles = [0, *angles, 0]
            for i, (name, values) in enumerate(df.iterrows()):
                values = values.to_list()
                values = [values[0], *values, values[0]]
                ax.plot(angles, values, color=colors[i], linewidth=3, linestyle='solid')
                ax.fill(angles, values, color=colors[i], alpha=0.2, label='_nolegend_')


            if legend:
                ax.legend(df.index.to_list(), loc='upper left', bbox_to_anchor=(-0.2, 1),
                          ncol=1, fancybox=True, shadow=True, fontsize=20,
                          frameon=True, framealpha=1, facecolor="white", edgecolor="black")
            plt.tight_layout()
            plt.savefig(f"{save_loc}.png")
            plt.savefig(f"{save_loc}.pdf")
            plt.show()


        def plot_task2(file_dir):
            scores = get_task2_scores(file_dir)

            scores_auroc = scores.set_index("name").iloc[:, 1:16:3]
            t2A_ticks = [0.7, 0.75, 0.8, 0.85, 0.9, ]
            t2B_ticks = [0.4, 0.45, 0.5, 0.55, 0.6, ]
            make_spider(scores_auroc, t2A_ticks, save_loc=f"../graphs/t2{dataset_type}auroc")

            scores_oe = scores.set_index("name").iloc[:, 16:-1:3]
            t2A_ticks_oe = [0.5, 1, 1.5, 2, 2.5]
            t2B_ticks_oe = [0, 0.5, 1, 1.5, 2, ]
            make_spider(
                scores_oe,
                t2A_ticks_oe,
                save_loc=f"../graphs/t2{dataset_type}oe",
                ytick_bold=1,
                legend=False
            )
            # plot_multi_spider([scores_auroc, scores_oe], [t2A_ticks, t2A_ticks_oe])


        def plot_multi_C(file_dir, file_names):
            test_scores_df = get_task1_scores_from_txt(file_dir, file_names)
            test_scores_df = test_scores_df[["name", "lc", "c", "hc"]]
            test_scores_df["type"] = "Test"

            histories = get_histories()
            values_df = {name: history.get(f"{type_name} C-Score") for name, history in histories}

            values_df = pd.DataFrame(values_df)
            average_values = values_df.mean(axis=0)
            confidence_intervals = values_df.quantile(q=[0.025, 0.975])
            wandb_scores_df = pd.concat([average_values, confidence_intervals.transpose()], axis=1)
            wandb_scores_df = wandb_scores_df.clip(upper=0.995)
            wandb_scores_df = wandb_scores_df.reset_index()
            wandb_scores_df = wandb_scores_df.rename(columns={0.0: "c", 0.025: "lc", 0.975: "hc", "index": "name"})
            wandb_scores_df["type"] = "Val"

            wandb_scores_df["name_ordered"] = pd.Categorical(wandb_scores_df["name"], categories=test_scores_df["name"],
                                                             ordered=True)
            wandb_scores_df = wandb_scores_df.sort_values("name_ordered")

            combined_df = pd.concat([test_scores_df, wandb_scores_df])

            fig, ax = plt.subplots(figsize=(10, 8))
            fig.subplots_adjust(left=0.28)
            colors = ["tab:blue", "tab:orange"]
            for offset, score_df in enumerate([wandb_scores_df, test_scores_df]):

                y_pos = np.array(range(len(score_df["name"])))

                for i, interval in enumerate(score_df[["lc", "hc"]].values):
                    xerr = (interval[1] - interval[0]) / 2
                    if score_df["c"].iloc[i] + xerr >= 0.995:
                       xerr = 0.995 - score_df["c"].iloc[i]

                    ax.errorbar(
                        score_df["c"].iloc[i], y_pos[i] - 0.1 + offset * 0.2, xerr=xerr, elinewidth=4,
                        capsize=10, linewidth=4, markersize=10, color=colors[offset],
                    )
                ax.scatter(score_df["c"], y_pos - 0.1 + offset * 0.2, marker="D", color=colors[offset], linewidth=8)

            ax.set_xlabel('C-Index', fontweight='bold', fontsize=22)
            ax.set_ylabel('Submitted run', fontweight='bold', fontsize=22)
            ax.set_xlim((0, 1))
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20, ticks=y_pos, labels=list(score_df["name"]))
            # ax.set_title(f'Dataset {self.dataset_name[-1]}', fontsize=20, fontweight='bold')
            # ax.legend(["Val", "Test"], fontsize=20, loc="lower left")
            ax.legend(["Val", "Test"], loc='lower left',
                      ncol=1, fancybox=True, shadow=True, fontsize=20,
                      frameon=True, framealpha=1, facecolor="white", edgecolor="black")
            # plt.grid()
            plt.tight_layout()
            plt.savefig(f"../graphs/CIndex_{dataset_type}comparison.png")
            plt.savefig(f"../graphs/CIndex_{dataset_type}comparison.pdf")
            plt.show()

        def get_histories():
            datasetA_runs = ["0pqogcd9", "pefhr65z", "bmxri49g", "eg47otd5", "tyke6gd4", "9ckubkgv", "i02w78l0",
                             "tcuu3nlg", "gpibc3q4"]
            datasetB_runs = ["aksefkpg", "dd6u6bby", "4j1qq5bt", "o1dl9vrp", "doh6ghxb", "38z0p36o", "3sojqzwl",
                             "25vl5d6k", "xqqj2y75"]

            current_dataset_runs = datasetA_runs if dataset_type == "A" else datasetB_runs

            api = wandb.Api()
            entity, project = 'mrhanzl', self.project
            runs = api.runs(entity + "/" + project)

            histories = []
            for run in runs:
                if run.id not in current_dataset_runs:
                    continue
                history = run.history()
                histories.append((run.name, history))

            def sort_f(hist_entry):
                return hist_entry[1][f"{type_name} C-Score Average"].iloc[-1]

            histories.sort(key=sort_f, reverse=True)

            return histories

        def plot_wandb():
            datasetA_runs = ["0pqogcd9", "pefhr65z", "bmxri49g", "eg47otd5", "tyke6gd4", "9ckubkgv", "i02w78l0", "tcuu3nlg", "gpibc3q4"]
            datasetB_runs = ["aksefkpg", "dd6u6bby", "4j1qq5bt", "o1dl9vrp", "doh6ghxb", "38z0p36o", "3sojqzwl", "25vl5d6k", "xqqj2y75"]

            dataset_type = self.dataset_name[-1]

            type_name = "Val"
            save_name = f"{type_name}C_{dataset_type}"

            api = wandb.Api()
            entity, project = 'mrhanzl', self.project
            runs = api.runs(entity + "/" + project)

            histories = []
            for run in runs:

                if run.id not in datasetA_runs:
                    continue
                history = run.history()
                histories.append((run.name, history))

            def sort_f(hist_entry):
                return hist_entry[1][f"{type_name} C-Score Average"].iloc[-1]
            histories.sort(key=sort_f, reverse=True)

            fig, ax = plt.subplots(figsize=(10, 8))
            # fig.subplots_adjust(right=0.9)
            # fig.subplots_adjust(left=0.28)
            for name, history in histories:
                if "minval" in name.lower():
                    continue
                step = history.get("_step")
                avg = history.get(f"{type_name} C-Score Average")
                std = history.get(f"{type_name} C-Std")
                ax.plot(step, avg, label=name, marker='|', linewidth=4, markersize=16)
                ax.fill_between(step, avg - std, avg + std, alpha=0.2)

            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            ax.set_xlabel('Steps', fontweight='bold', fontsize=22)
            ax.set_ylabel('C-Score', fontweight='bold', fontsize=22)
            ax.set_xlim((0, 100))
            ax.set_ylim((0.4, 0.85))
            # ax.set_xticks(np.arange(0, 1.01, step=0.05))
            # ax.set_title(f'{"Validation" if type_name == "Val" else "Train"} C-Index,'
            #              f' Dataset {self.dataset_name[-1]}', fontsize=18, fontweight='bold')
            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0),
                      ncol=2, fancybox=True, shadow=True, fontsize=20,
                      frameon=True, framealpha=1, facecolor="white", edgecolor="black")
            if "minval" in save_name:  # hotfix
                for i, text in enumerate(legend.texts):
                    legend.texts[i].set_text(text.get_text().removesuffix(" - MinVal"))

            plt.tight_layout()
            # plt.savefig(f"../graphs/{save_name}.png")
            plt.savefig(f"../graphs/{save_name}.pdf")
            # plt.savefig(f"../graphs/{save_name}.svg")
            # plt.savefig(f"../graphs/{save_name}.eps")
            plt.show()

        def plot_wandb_box():
            histories = get_histories()
            values_df = {name: history.get(f"{type_name} C-Score") for name, history in histories}
            values_df = pd.DataFrame(values_df)

            values_df.columns = [name.split(" - ")[0] for name in values_df.columns.values]

            fig, ax = plt.subplots(figsize=(10, 3))
            sns.boxplot(values_df, orient="h", hue="type", linewidth=2, fliersize=10)

            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            ax.set_ylabel('Method', fontweight='bold', fontsize=22)
            ax.set_xlabel('C-Index', fontweight='bold', fontsize=22)
            ax.set_xlim((0, 1))
            # ax.set_ylim((0.4, 0.85))
            # ax.set_xticks(np.arange(0, 1.01, step=0.05))
            # ax.set_title(f'{"Validation" if type_name == "Val" else "Train"} C-Index,'
            #              f' Dataset {self.dataset_name[-1]}', fontsize=18, fontweight='bold')
            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0),
            #                    ncol=2, fancybox=True, shadow=True, fontsize=20,
            #                    frameon=True, framealpha=1, facecolor="white", edgecolor="black")
            # if "minval" in save_name:  # hotfix
            #     for i, text in enumerate(legend.texts):
            #         legend.texts[i].set_text(text.get_text().removesuffix(" - MinVal"))

            plt.tight_layout()
            plt.savefig(f"../graphs/{save_name}.png")
            plt.savefig(f"../graphs/{save_name}.pdf")
            # plt.savefig(f"../graphs/{save_name}.svg")
            # plt.savefig(f"../graphs/{save_name}.eps")
            plt.show()

        # plot_CIndex(file_names)
        # plot_wandb_box()
        # plot_multi_C(task1_file_dir, task1_file_names)
        plot_task2(task2_file_dir)


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

    DATASET = "datasetA"
    DATASET_DIR = f"../data/{DATASET}_train"
    ID_FEAT = "patient_id"

    pipeline = IDPPPipeline(DATASET_DIR, DATASET, ID_FEAT, DEFAULT_RANDOM_SEED)
    pipeline.plot_submission_results()
    # pipeline.run()
    # pipeline.param_sweep()


if __name__ == "__main__":
    main()
