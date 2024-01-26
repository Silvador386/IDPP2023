import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from ioutils import load_dfs_from_files_in_dir, save_predictions, load_df_from_file, filenames_in_folder
from dataset import IDDPDataset
from preprocessing import preprocess, fastai_preproccess_dataset, y_to_struct_array
from evaluation import evaluate_c, evaluate_cumulative, plot_coef_, wrap_c_scorer
from survEstimators import init_surv_estimators, SurvTraceWrap, AvgEnsemble
from sklearn.model_selection import ShuffleSplit
from wandbsetup import setup_wandb, launch_sweep
import wandb
from sklearn import set_config

set_config(display="text")  # displays text representation of estimators


class IDPPPipeline:
    OUTPUT_DIR = "../out"

    def __init__(
            self,
            config: dict
    ):
        self.config = config

        self.dataset_name = config.get("dataset_name")
        self.id_feature_name = config.get("id_feature_name")
        self.seed = config.get("seed", 777)
        self.num_iter = config.get("num_iter", 10)
        self.train_size = config.get("train_size", 0.8)
        self.n_estimators = config.get("n_estimators", 100)

        self.dataset = None
        self.estimators = None

    def setup(self):
        dataset_dirs = [f"../data/{self.dataset_name}_train", f"../data/{self.dataset_name}_test"]

        self.dataset = IDDPDataset(self.dataset_name, self.id_feature_name, self.seed)
        self.dataset.load_data(*dataset_dirs)

        X, y, y_struct = self.dataset.get_train_data()
        # self.X_test, self.y_test = self.dataset.get_test_data()

        self.estimators = init_surv_estimators(self.seed, X, y, self.n_estimators)

        self.team_shortcut_t1 = "uwb_T1{}_{}"
        self.team_shortcut_t2 = "uwb_T2{}_{}"

    def run(self):
        self.setup()
        X, y, y_struct = self.dataset.get_train_data()
        best_accs, avg_acc, best_est = [], [], []
        for name, models in self.estimators.items():
            # self.wandb_run = setup_wandb(self.config, name, self.dataset )
            print(f"Estimator: {name}")
            train_c_scores, val_c_scores, best_acc, best_estimator = self.run_n_times(models, self.dataset, num_iter=self.num_iter)
            best_accs.append(best_acc)
            avg_acc.append(np.average(val_c_scores))
            best_est.append(best_estimator)
            print(f"Train ({self.num_iter}iter) c-score: {np.average(train_c_scores)} ({np.std(train_c_scores)})")
            print(f"Val   ({self.num_iter}iter) c-score: {np.average(val_c_scores)} ({np.std(val_c_scores)})")

            if name == "CGBSA":
                plot_coef_(best_estimator, X)

            self.predict(best_estimator, X, y_struct, save=False)

            if name != "SurvTRACE":
                self.predict_cumulative(best_estimator, X, y_struct, save=False)

            # if name == "SurvTRACE_cumulative":
            #     self.predict_cumulative(best_estimator, self.X_test, save=True)

            self.wandb_run.finish()

        best_estimator_index = np.array(avg_acc).argmax()
        best_estimator = best_est[best_estimator_index]
        best_est_name = list(self.estimators.keys())[best_estimator_index]

        best_est_name = "AvgEnsemble"

        self.team_shortcut_t1 = self.team_shortcut_t1.format(self.dataset_name[-1].lower(), best_est_name)
        self.team_shortcut_t2 = self.team_shortcut_t2.format(self.dataset_name[-1].lower(), best_est_name)

        self.predict(best_estimator, X, y_struct, save=False)
        # self.predict(best_estimator, self.X_test, save=True)

        # if best_est_name != "SurvTRACE":
        #     self.predict_cumulative(best_estimator, self.X, (self.y_struct, self.y_struct), save=False)
        #     self.predict_cumulative(best_estimator, self.X_test, save=True)

        ensemble = AvgEnsemble(best_est)
        self.wandb_run = setup_wandb(self.config, model_name="EnsembleAvg", dataset=self.dataset)
        self.run_n_times(ensemble, self.dataset, 100)
        self.predict(ensemble, X, y_struct, save=False)
        # self.predict(ensemble, self.X_test, save=True)
        # self.predict_cumulative(ensemble, self.X, (self.y_struct, self.y_struct), save=False)
        # self.predict_cumulative(ensemble, self.X_test, save=True)
        self.wandb_run.finish()

    def run_model(self, model, dataset: IDDPDataset, random_state: int) -> tuple[list, list, list]:

        X, y_df, y_struct = dataset.get_train_data()
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

    def run_n_times(self, model, dataset: IDDPDataset, num_iter: int = 5) -> tuple:
        train_c_scores, val_c_scores, fitted_models = [], [], []
        for i in range(num_iter):
            error_flag = True
            while error_flag:
                try:
                    train_c_score, val_c_score, fitted_model = self.run_model(model, dataset, random_state=random.randint(0, 2**10))
                    train_c_scores += train_c_score
                    val_c_scores += val_c_score
                    fitted_models += fitted_model
                    error_flag = False

                except ValueError:
                    print("Error")
            # if i % 5 == 0 or (i+1) == num_iter:
            # self.wandb_run.log({f"Num Iter": i+1,
            #                     f"Train C-Score Average": np.average(train_c_scores),
            #                     f"Val C-Score Average": np.average(val_c_scores),
            #                     f"Train C-Score": train_c_score[0],
            #                     f"Val C-Score": val_c_score[0],
            #                     f"Train C-Std": np.std(train_c_scores),
            #                     f"Val C-Std": np.std(val_c_scores)
            #                     })

        best_acc, best_estimator = (np.max(np.array(val_c_scores)), fitted_models[np.array(val_c_scores).argmax()])
        return np.array(train_c_scores), np.array(val_c_scores), best_acc, best_estimator

    def predict(self, best_model, X: pd.DataFrame, y: list = None, save: bool = False) -> pd.DataFrame:

        c_score, predictions = evaluate_c(best_model, X, y)

        pred_output = {self.id_feature_name: X.index,
                       "predictions": predictions,
                       "run": self.team_shortcut_t1}

        print(best_model.__class__.__name__)
        print(f"Predictions C-Index Resized:", c_score)
        pred_df = pd.DataFrame(pred_output)

        if save:
            save_predictions(self.OUTPUT_DIR, f"{self.team_shortcut_t1}.txt", pred_df)
        return pred_df

    def predict_cumulative(self, best_model, X: pd.DataFrame, y: list = None, save: bool = False) -> pd.DataFrame:
        time_points = [2, 4, 6, 8, 10]
        auc_scores, predictions = evaluate_cumulative(best_model, X, y, time_points=time_points,
                                                      plot=True)

        pred_output = {self.id_feature_name: X.index,
                       "2years": predictions[:, 0],
                       "4years": predictions[:, 1],
                       "6years": predictions[:, 2],
                       "8years": predictions[:, 3],
                       "10years": predictions[:, 4],
                       "run": self.team_shortcut_t2}
        print("AUC Scores whole", auc_scores)
        pred_df = pd.DataFrame(pred_output)
        if save:
            save_predictions(self.OUTPUT_DIR, f"{self.team_shortcut_t2}.txt", pred_df)

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

        def get_task1_scores_from_txt(file_dir: str, file_names: list[str]) -> pd.DataFrame:
            score_data = [load_df_from_file(f"{file_dir}{file_name}") for file_name in file_names if f"T1{self.dataset_name[-1].lower()}" in file_name]
            score_df = pd.concat(score_data)
            score_df[0] = ["-".join(name.removesuffix(".txt").split("_")[-2:]) if "minVal" in name else
                           name.removesuffix(".txt").split("_")[-1] for name in score_df[0]]
            score_df.columns = ["name", "lc", "c", "hc", "count"]
            score_df = score_df.sort_values(by="c", ascending=True)

            name_map = {'SurvTRACE-minVal': "SurvTRACE - MinVal", 'survRF': "Random Forest", 'CGBSA': "CGBSA", 'survRFmri': "Random Forest MRI", 'survGB': "Gradient Boosting",
            "AvgEnsemble": 'Ensemble Avg.', 'AvgEnsemble-minVal': 'Ensemble Avg. - MinVal', 'survGB-minVal': 'Gradient Boosting - MinVal', 'SurvTRACE': "SurvTRACE"}
            score_df["name"].replace(name_map, inplace=True)
            return score_df

        def get_task2_scores(file_dir: str) -> pd.DataFrame:
            file_names = filenames_in_folder(file_dir)
            score_data = [load_df_from_file(f"{file_dir}{file_name}") for file_name in file_names if
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

        def make_spider(df: pd.DataFrame, yticks, save_loc, ytick_bold: int = None, legend=True):
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
                if i == -2:
                    ax.plot(angles, values, color=colors[i], linewidth=5, linestyle='solid')
                    ax.fill(angles, values, color=colors[i], alpha=0.5, label='_nolegend_', hatch="x")
                else:
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
            make_spider(scores_auroc, t2A_ticks, save_loc=f"../graphs/t2{dataset_type}auroc",
                        # ytick_bold=2
                        )

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

            plt.xticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1], fontsize=20)

            labels = ax.get_xticklabels()
            labels[3].set_fontweight('bold')
            labels[3].set_fontsize(20)

            gridlines = ax.xaxis.get_gridlines()
            gridlines[3].set_color("k")
            gridlines[3].set_linewidth(3)
            gridlines[3].set_linestyle("--")

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

        # plot_CIndex(file_names)
        # plot_wandb_box()
        # plot_multi_C(task1_file_dir, task1_file_names)
        plot_task2(task2_file_dir)


    def run_ensemble(self):
        from sksurv.meta import EnsembleSelectionRegressor, Stacking

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

    config = {
        "dataset_name": "datasetA",
        "id_feature_name": "patient_id",
        "num_iter": 100,
        "train_size": 0.8,
        "seed": DEFAULT_RANDOM_SEED,
    }

    pipeline = IDPPPipeline(config)
    # pipeline.plot_submission_results()
    pipeline.run()
    # pipeline.param_sweep()


if __name__ == "__main__":
    main()
