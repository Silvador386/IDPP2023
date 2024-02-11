import copy
import numpy as np
from threading import Lock

from sksurv.ensemble import RandomSurvivalForest, ComponentwiseGradientBoostingSurvivalAnalysis, \
    GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastKernelSurvivalSVM, HingeLossSurvivalSVM, MinlipSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis

from load_survtrace import load_data
from survtrace.survtrace.evaluate_utils import Evaluator
from survtrace.survtrace.utils import set_random_seed
from survtrace.survtrace.model import SurvTraceSingle
from survtrace.survtrace.train_utils import Trainer
from survtrace.survtrace.config import STConfig


def init_surv_estimators(seed, X, y_df, n_estimators=100):
    rsf = RandomSurvivalForest(n_estimators=300, max_depth=6, min_samples_split=10, min_samples_leaf=3,
                               oob_score=True, n_jobs=6, random_state=seed)
    gbs = GradientBoostingSurvivalAnalysis(n_estimators=500, learning_rate=0.5, max_depth=3, min_samples_split=4,
                                           min_samples_leaf=1, subsample=0.5, dropout_rate=0.2, random_state=seed)
    # gbs = GradientBoostingSurvivalAnalysis(random_state=seed)
    # msa = MinlipSurvivalAnalysis()
    cgb = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=300, learning_rate=0.5, dropout_rate=0.2,
                                                        subsample=0.75, random_state=seed)
    # cgb = ComponentwiseGradientBoostingSurvivalAnalysis(random_state=seed)

    cox = CoxPHSurvivalAnalysis()
    surv_trace = SurvTraceWrap(seed, X, y_df, instance_order=0, cumulative=False)
    surv_trace_cumulative = SurvTraceWrap(seed, X, y_df, instance_order=0, cumulative=True)

    estimators = {
        # "RandomForest": rsf,
        # "GradientBoost": gbs,
        # # "MinlipSA": msa,
        # "CGBSA": cgb,
        # "Cox": cox
        "SurvTRACE": surv_trace,
        # "SurvTRACE_cumulative": surv_trace_cumulative,
    }

    return estimators


def init_model(seed, **params):
    # rsf = RandomSurvivalForest(n_jobs=6, random_state=seed, **params)
    # gbs = GradientBoostingSurvivalAnalysis(random_state=seed, **params)
    cgb = ComponentwiseGradientBoostingSurvivalAnalysis(random_state=seed, **params)
    return cgb


class AvgEnsemble:
    def __init__(self, estimators):
        self.estimators = estimators
        self.num_est = len(estimators)
        self.averaging_coeffs = [0.25, 0.5, 0.25]

    def predict(self, X):
        predictions = np.array([estimator.predict(X).reshape(-1) for estimator in self.estimators]).T
        predictions = (predictions - np.average(predictions, axis=0)) / np.std(predictions, axis=0)
        averaging = np.average(predictions, axis=1)
        # averaging = sum([coef*predictions[:, i] for i, coef in enumerate(self.averaging_coeffs)]) # max
        maxing = np.max(predictions, axis=1)
        return averaging

    def predict_cumulative(self, X):
        time_points = [2, 4, 6, 8, 10]
        model_predictions = []
        for model in self.estimators:
            pred_surv = model.predict_cumulative_hazard_function(X)
            predictions = []
            for i, surv_func in enumerate(pred_surv):
                predictions.append(surv_func(time_points))

            predictions = (np.array(predictions) - np.average(predictions, axis=1).reshape(-1, 1)) / np.std(predictions,
                                                                                                            axis=1).reshape(
                -1, 1)
            model_predictions.append(predictions)

        model_predictions = np.array(model_predictions)
        averaging = np.average(model_predictions, axis=0)
        maxing = np.max(model_predictions, axis=0)
        # averaging = [coef * model_predictions[i] for i, coef in enumerate(self.averaging_coeffs)]
        return averaging

    def fit(self, X, y_df):
        pass


class SurvTraceWrap():
    hparams = {
        'batch_size': 64,
        'weight_decay': 0.0000284,
        'learning_rate': 0.006157,
        'epochs': 100,
    }

    def __init__(
            self,
            seed,
            instance_order: int,
            cumulative=False,
            **kwargs
    ):

        self.seed = seed
        self.STConfig = copy.deepcopy(STConfig)
        self.STConfig['data'] = 'idpp'
        self.STConfig['seed'] = self.seed

        self.STConfig.update(checkpoint=f"./survtrace/checkpoints/iddp_{instance_order:03d}.pt")

        if cumulative:
            self.STConfig["times"] = [2, 4, 6, 8, 10]
        if kwargs:
            self.STConfig.update(**kwargs)
            self.hparams = {
                'batch_size': kwargs['batch_size'],
                'weight_decay': kwargs['weight_decay'],
                'learning_rate': kwargs['learning_rate'],
                'epochs': kwargs['epochs'],
            }
        else:
            self.hparams = SurvTraceWrap.hparams

    def fit(self, X, y_df, train_idx, val_idx):
        hparams = self.hparams
        df, df_train, df_y_train, df_val, df_y_val = load_data(self.STConfig, X, y_df, train_idx,
                                                                                   val_idx)
        self.model = SurvTraceSingle(self.STConfig)
        self.trainer = Trainer(self.model)

        train_loss, val_loss = self.trainer.fit((df_train, df_y_train), (df_val, df_y_val),
                                                batch_size=hparams['batch_size'],
                                                epochs=hparams['epochs'],
                                                learning_rate=hparams['learning_rate'],
                                                weight_decay=hparams['weight_decay'], )

        scores = []
        for X_pred, indexes in zip([df_train, df_val], [train_idx, val_idx]):
            preds = self.model.predict(X_pred, batch_size=64)
            preds = preds.cpu()
            scores.append(concordance_index_censored(y_df.iloc[indexes]["outcome_occurred"].astype(bool),
                                                     y_df.iloc[indexes]["outcome_time"], preds[:, -1])[0])

        return self, scores[0], scores[1]

    def predict(self, X):
        predictions = self.model.predict(X, batch_size=self.hparams["batch_size"])
        predictions = predictions.cpu().numpy()
        return predictions[:, -1]

    def predict_cumulative(self, X):
        predictions = self.model.predict(X, batch_size=self.hparams["batch_size"])
        predictions = predictions.cpu().numpy()
        # predictions += np.min(predictions)
        cumulative_preds = np.zeros(predictions.shape)
        # cumulative_preds[:, 0] = predictions[:, 0]
        for i in range(predictions.shape[1]):
            cumulative_preds[:, i:] += predictions[:, i].reshape(-1, 1)
        return predictions[:, :-1]
