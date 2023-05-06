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
    rsf = RandomSurvivalForest(n_estimators=n_estimators, n_jobs=6, random_state=seed)
    gbs = GradientBoostingSurvivalAnalysis(n_estimators=n_estimators, random_state=seed)
    # msa = MinlipSurvivalAnalysis()
    cgb = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=n_estimators, random_state=seed)
    cox = CoxPHSurvivalAnalysis()
    surv_trace = SurvTraceWrap(seed, X, y_df)

    estimators = {
        # "SurvTRACE": surv_trace,
        "RandomForest": rsf,
        "GradientBoost": gbs,
        # "MinlipSA": msa,
        "CGBSA": cgb,
        # "Cox": cox
    }

    return estimators


def init_model(seed, **params):
    rsf = RandomSurvivalForest(n_jobs=6, random_state=seed, **params)
    estimators = {
        "RandomForest": rsf,
    }

    return rsf

class SurvTraceWrap:
    hparams = {
        'batch_size': 64,
        'weight_decay': 1e-4,
        'learning_rate': 1e-3,
        'epochs': 20,
    }

    def __init__(self, seed, X, y_df):
        pass
        # # Update config
        # df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_data(STConfig, X, y_df)
        # # get model
        # self.model = SurvTraceSingle(STConfig)
        # self.trainer = Trainer(self.model)

    def fit(self, X, y_df, train_idx, val_idx, test_c=True):
        STConfig['data'] = 'idpp'
        STConfig['seed'] = 2021
        hparams = SurvTraceWrap.hparams
        df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_data(STConfig, X, y_df, train_idx,
                                                                                   val_idx)
        self.model = SurvTraceSingle(STConfig)
        self.trainer = Trainer(self.model)

        train_loss, val_loss = self.trainer.fit((df_train, df_y_train), (df_val, df_y_val),
                                                batch_size=hparams['batch_size'],
                                                epochs=hparams['epochs'],
                                                learning_rate=hparams['learning_rate'],
                                                weight_decay=hparams['weight_decay'], )

        scores = []
        for X_pred, indexes in zip([df_train, df_val], [train_idx, val_idx]):
            preds = self.model.predict(X_pred, batch_size=64)
            scores.append(concordance_index_censored(y_df.iloc[indexes]["outcome_occurred"].astype(bool),
                                                     y_df.iloc[indexes]["outcome_time"], preds[:, -1])[0])
            print(preds)
        print(f"SurfTrace Scores: {scores}")
        return self.model, scores[0], scores[1]

    def predict(self, X):
        predictions = self.model.predict(X, batch_size=SurvTraceWrap.hparams["batch_size"])
        return predictions[:, -1]


def run_survtrace(seed, X, y_df, train_idx=None, test_idx=None):
    STConfig['data'] = 'idpp'
    STConfig['seed'] = seed

    hparams = {
        'batch_size': 64,
        'weight_decay': 1e-4,
        'learning_rate': 1e-3,
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
