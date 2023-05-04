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


def init_surv_estimators(seed, n_estimators=100):
    rsf = RandomSurvivalForest(n_estimators=n_estimators, n_jobs=6, random_state=seed)
    gbs = GradientBoostingSurvivalAnalysis(n_estimators=n_estimators, random_state=seed)
    # msa = MinlipSurvivalAnalysis()
    cgb = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=n_estimators, random_state=seed)
    cox = CoxPHSurvivalAnalysis()

    estimators = {"RandomForest": rsf,
                  "GradientBoost": gbs,
                  # "MinlipSA": msa,
                  "CGBSA": cgb,
                  # "Cox": cox
                  # "SurvTRACE": "SurvTRACE",
                  }

    return estimators


def run_survtrace(seed, merged_df, X, y_df, train_idx, test_idx):
    STConfig['data'] = 'idpp'
    STConfig['seed'] = seed

    hparams = {
        'batch_size': 64,
        'weight_decay': 1e-4,
        'learning_rate': 1e-3,
        'epochs': 40,
    }

    # load data
    df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_data(STConfig, merged_df, X, y_df, train_idx, test_idx)

    # get model
    model = SurvTraceSingle(STConfig)

    # initialize a trainer
    trainer = Trainer(model)
    train_loss, val_loss = trainer.fit((df_train, df_y_train), (df_val, df_y_val),
                                        batch_size=hparams['batch_size'],
                                        epochs=hparams['epochs'],
                                        learning_rate=hparams['learning_rate'],
                                        weight_decay=hparams['weight_decay'], )
    # evaluate model
    evaluator = Evaluator(df, df_train.index)
    evaluator.eval(model, (df_val, df_y_val))
    print("done")

    scores = []
    for X_pred, y_pred in zip([df_train, df_val], [df_y_train, df_y_val]):
        preds = model.predict(X_pred, batch_size=64)
        scores.append(concordance_index_censored(y_pred["event"].astype(bool), y_pred["duration"], preds[:, -1]))

    return model, scores[0], scores[1]
