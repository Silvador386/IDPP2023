import copy
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
    rsf = RandomSurvivalForest(n_estimators=100, max_depth=6, min_samples_split=10, min_samples_leaf=3,
                               n_jobs=6, random_state=seed)
    gbs = GradientBoostingSurvivalAnalysis(n_estimators=500, learning_rate=0.5, max_depth=3, min_samples_split=4,
                                           min_samples_leaf=1, subsample=0.5, dropout_rate=0.25, random_state=seed)
    # msa = MinlipSurvivalAnalysis()
    cgb = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=300, learning_rate=0.1, subsample=0.2, random_state=seed)
    cox = CoxPHSurvivalAnalysis()
    surv_trace = SurvTraceWrap(seed, X, y_df, cumulative=False)
    surv_trace_cumulative = SurvTraceWrap(seed, X, y_df, cumulative=True)

    estimators = {
        # "RandomForest": rsf,
        # "GradientBoost": gbs,
        # "MinlipSA": msa,
        # "CGBSA": cgb,
        # "Cox": cox
        "SurvTRACE": surv_trace,
        "SurvTRACE_cumulative": surv_trace_cumulative,
    }

    return estimators


def init_model(seed, **params):
    # rsf = RandomSurvivalForest(n_jobs=6, random_state=seed, **params)
    # gbs = GradientBoostingSurvivalAnalysis(random_state=seed, **params)
    cgb = ComponentwiseGradientBoostingSurvivalAnalysis(random_state=seed, **params)
    return cgb


class SurvTraceWrap:
    model_counter = 0
    hparams = {
        'batch_size': 64,
        'weight_decay': 0.0000284,
        'learning_rate': 0.006157,
        'epochs': 30,
    }

    def __init__(self, seed, X, y_df, wandb_run=None, cumulative=False, **params):
        self.seed = seed
        self.STConfig = copy.deepcopy(STConfig)
        self.STConfig['data'] = 'idpp'
        self.STConfig['seed'] = self.seed
        if cumulative:
            self.STConfig["times"] = [2, 4, 6, 8, 10]
        if params:
            self.STConfig.update(**params)
            self.hparams = {'batch_size': params['batch_size'],
                            'weight_decay': params['weight_decay'],
                            'learning_rate': params['learning_rate'],
                            'epochs': params['epochs'],
                            }
        else:
            self.hparams = SurvTraceWrap.hparams
        if wandb_run:
            self.wandb_run = wandb_run
        df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_data(self.STConfig, X, y_df)

    def fit(self, X, y_df, train_idx, val_idx):
        self.model_count = SurvTraceWrap.model_counter
        SurvTraceWrap.model_counter += 1
        self.STConfig.update(checkpoint=f"./survtrace/checkpoints/iddp_{self.model_count:03d}.pt"),
        hparams = self.hparams
        df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_data(self.STConfig, X, y_df, train_idx,
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

        # if self.wandb_run is not None:
        #     for train_loss, val_loss in zip(train_loss, val_loss):
        #         self.wandb_run.log({f"Train Loss": train_loss,
        #                             f"Val Loss": val_loss,
        #                             })
        print(f"SurfTrace Scores: {scores}")
        return self, scores[0], scores[1]

    def predict(self, X):
        predictions = self.model.predict(X, batch_size=self.hparams["batch_size"])
        predictions = predictions.cpu().numpy()
        return predictions[:, -1]

    def predict_multiple(self, X):
        predictions = self.model.predict(X, batch_size=self.hparams["batch_size"])
        predictions = predictions.cpu().numpy()
        return predictions[:, :-1]


def run_survtrace(seed, X, y_df, train_idx=None, test_idx=None):
    STConfig['data'] = 'idpp'
    STConfig['seed'] = seed

    hparams = {
        'batch_size': 64,
        'weight_decay': 0.0009008,
        'learning_rate': 0.001429,
        'epochs': 40,
    }

    # load data
    df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_data(STConfig, X, y_df, train_idx, test_idx)

    # get model
    model = SurvTraceSingle(STConfig)

    # initialize a trainer
    trainer = Trainer(model)
    train_loss, val_loss = trainer.fit((df_train, df_y_train), (df_val, df_y_val),
                                       batch_size=hparams['batch_size'],
                                       epochs=hparams['epochs'],
                                       learning_rate=hparams['learning_rate'],
                                       weight_decay=hparams['weight_decay'], )

    evaluator = Evaluator(df, df_train.index)
    evaluator.eval(model, (df_test, df_y_test))
    print("done")

    scores = []
    for X_pred, y_pred in zip([df_train, df_val], [df_y_train, df_y_val]):
        preds = model.predict(X_pred, batch_size=64)
        scores.append(concordance_index_censored(y_pred["event"].astype(bool), y_pred["duration"], preds[:, -1]))

    return model, scores[0], scores[1]
