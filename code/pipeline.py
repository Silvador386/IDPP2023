import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tqdm

from ioutils import load_dfs_from_files_in_dir, save_predictions, load_df_from_file, filenames_in_folder
from dataset import IDDPDataset
from preprocessing import preprocess, fastai_preproccess_dataset, y_to_struct_array
from evaluation import get_c_score, evaluate_cumulative_c_score, plot_coef_, wrap_c_scorer, \
    resize_pred_scores_by_order
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
        self.num_iter = config.get("num_iter", 1)
        self.train_size = config.get("train_size", 0.8)
        self.n_estimators = config.get("n_estimators", 100)

        self.dataset = None
        self.estimators = None

    def setup(self):
        train_dir_path = f"../data/{self.dataset_name}_train"
        test_dataset_dirs = f"../data/{self.dataset_name}_test"

        self.dataset = IDDPDataset(self.dataset_name, self.id_feature_name, self.seed)
        self.dataset.load_data(train_dir_path=train_dir_path, test_dir_path=test_dataset_dirs)

        X, y, y_struct = self.dataset.get_train_data()
        # self.X_test, self.y_test = self.dataset.get_test_data()

        self.estimators = init_surv_estimators(self.seed, X, y, self.n_estimators)

    def run(self):
        self.setup()
        X, y, y_struct = self.dataset.get_train_data()
        X_test, y_test, y_test_struct = self.dataset.get_test_data()
        best_accs, avg_acc, best_estimators = [], [], []
        for name, models in self.estimators.items():
            print(f"Estimator: {name}")
            train_c_scores, val_c_scores, best_acc, best_estimator = self.run_folds(models, self.dataset, self.num_iter)
            best_accs.append(best_acc)
            avg_acc.append(np.average(val_c_scores))
            best_estimators.append(best_estimator)
            print(f"Train ({self.num_iter}iter) c-score: {np.average(train_c_scores)} ({np.std(train_c_scores)})")
            print(f"Val   ({self.num_iter}iter) c-score: {np.average(val_c_scores)} ({np.std(val_c_scores)})")

            if name == "CGBSA":
                plot_coef_(best_estimator, X)

            if name != "SurvTRACE":
                self.predict_cumulative(best_estimator, X_test, y_struct, y_test_struct)

        best_estimator_index = np.array(avg_acc).argmax()
        best_estimator = best_estimators[best_estimator_index]
        self.predict_and_get_c_score(best_estimator, X_test, y_test_struct)

        self.run_ensemble(best_estimators)

    def run_ensemble(self, estimator_models):
        ensemble = AvgEnsemble(estimator_models)
        self.run_folds(ensemble, self.dataset, 100)


    def run_folds(self, model, dataset: IDDPDataset, iterations: int = 5) -> tuple:
        train_c_scores, val_c_scores, fitted_models = [], [], []

        wandb_run = setup_wandb(self.config, dataset=self.dataset, model_name=model.__class__.__name__)
        for i in tqdm.tqdm(range(iterations)):
            while True:
                try:
                    fitted_model, train_c_score, val_c_score = self.run_random_split(model, dataset, random_state=random.randint(0, 2 ** 10))
                    fitted_models.append(fitted_model)
                    train_c_scores.append(train_c_score)
                    val_c_scores.append(val_c_score)

                    wandb_run.log(
                        {
                            f"Train C-Score": train_c_score,
                            f"Val C-Score": val_c_score,
                            f"Avg. Train C-Score": np.average(train_c_scores),
                            f"Avg. Val C-Score": np.average(val_c_scores),
                        }
                    )
                    break

                except ValueError as e:
                    print(f"Error: {e}")

        best_acc, best_estimator = (np.max(np.array(val_c_scores)), fitted_models[np.array(val_c_scores).argmax()])
        if dataset.has_test_data():
            X_test, y_test, y_test_struct = dataset.get_test_data()
            test_c_score, prediction_scores = self.predict_and_get_c_score(best_estimator, X_test, y_test_struct)
            wandb_run.log(
                {
                    f"Test C-Score": test_c_score
                }
            )

        wandb_run.finish()

        return np.array(train_c_scores), np.array(val_c_scores), best_acc, best_estimator

    def run_random_split(self, model, dataset: IDDPDataset, random_state: int) -> tuple:

        X, y_df, y_struct = dataset.get_train_data()

        ss = ShuffleSplit(n_splits=1, train_size=self.train_size, random_state=random_state)

        for i, (train_idx, test_idx) in enumerate(ss.split(X, y_struct)):
            X_train, y_train, X_valid, y_valid = X.iloc[train_idx], y_struct[train_idx], X.iloc[test_idx], y_struct[test_idx]

            if model.__class__.__name__ == "SurvTraceWrap":
                # model = SurvTraceWrap(self.seed, X, y_df)
                model, train_c_score, val_c_score = model.fit(X, y_df, train_idx, test_idx)
            else:
                model.fit(X_train, y_train)
                train_c_score, _ = self.predict_and_get_c_score(model, X_train, y_train)
                val_c_score, _ = self.predict_and_get_c_score(model, X_valid, y_valid)

        return model, train_c_score, val_c_score

    def predict_and_get_c_score(
            self,
            model,
            X: pd.DataFrame,
            y: list
    ) -> tuple[float, pd.DataFrame]:
        prediction_scores = self.predict(model, X)
        c_score = get_c_score(prediction_scores, y)

        pred_output = {
            self.id_feature_name: X.index,
            "predictions": prediction_scores,
            "run": f"uwb_T1{self.dataset_name[-1]}_{model.__class__.__name__}"
        }
        pred_df = pd.DataFrame(pred_output)
        return c_score, pred_df

    @staticmethod
    def predict(model, X: pd.DataFrame) -> np.array:
        prediction_scores = model.predict(X)
        prediction_scores = resize_pred_scores_by_order(prediction_scores)
        return prediction_scores

    def predict_cumulative(
            self,
            model,
            X: pd.DataFrame,
            y_struct_train: list = None,
            y_struct_test: list = None,
    ) -> tuple[float, pd.DataFrame]:
        time_points = [2, 4, 6, 8, 10]
        auc_scores, predictions = evaluate_cumulative_c_score(
            model,
            X,
            y_struct_train=y_struct_train,
            y_struct_test=y_struct_test,
            time_points=time_points,
            plot=True
        )

        pred_output = {
            self.id_feature_name: X.index,
            "2years": predictions[:, 0],
            "4years": predictions[:, 1],
            "6years": predictions[:, 2],
            "8years": predictions[:, 3],
            "10years": predictions[:, 4],
            "run": f"uwb_T2{self.dataset_name[-1]}_{model.__class__.__name__}"
        }
        pred_df = pd.DataFrame(pred_output)

        return auc_scores, pred_df

    def param_sweep(self):
        sweep_id = launch_sweep(self.project, self.notes, self.config)
        wandb.agent(sweep_id, function=self.train_sweep_wrap)

    def train_sweep_wrap(self, ):
        with wandb.init():
            params = {**wandb.config}
            model = SurvTraceWrap(self.seed, self.X, self.y, wandb, **params)
            train_c_score, val_c_score, fitted_model = self.run_random_split(model, self.seed)
            wandb.log({f"Train C-Score": train_c_score[0],
                       f"Val C-Score": val_c_score[0],
                       })

    def _run_ensemble(self):
        from sksurv.meta import EnsembleSelectionRegressor, Stacking

        model = EnsembleSelectionRegressor([(name, est) for name, est in self.estimators.items()], scorer=wrap_c_scorer,
                                           n_jobs=10)

        self.wandb_run = setup_wandb(project=self.project, config=self.config, name="EnsembleSelection",
                                     notes=self.notes)
        print(f"Estimator: EnsembleSelection")
        train_c_scores, val_c_scores, best_acc, best_estimator = self.train_(model, num_iter=self.num_iter)
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
        "num_iter": 1,
        "train_size": 0.8,
        "seed": DEFAULT_RANDOM_SEED,

        "wandb_entity": "mrhanzl",
        "wandb_project": "IDPP-2024",
    }

    pipeline = IDPPPipeline(config)
    # pipeline.plot_submission_results()
    pipeline.run()
    # pipeline.param_sweep()


if __name__ == "__main__":
    main()
