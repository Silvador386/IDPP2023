from sksurv.ensemble import RandomSurvivalForest, ComponentwiseGradientBoostingSurvivalAnalysis, \
    GradientBoostingSurvivalAnalysis
from sksurv.svm import FastKernelSurvivalSVM, HingeLossSurvivalSVM, MinlipSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis

from survtrace.survtrace.dataset import load_data
from survtrace.survtrace.evaluate_utils import Evaluator
from survtrace.survtrace.utils import set_random_seed
from survtrace.survtrace.model import SurvTraceSingle
from survtrace.survtrace.train_utils import Trainer
from survtrace.survtrace.config import STConfig


def init_surv_estimators(seed, n_estimators=100):
    rsf = RandomSurvivalForest(n_estimators=n_estimators, min_samples_leaf=7, n_jobs=6, random_state=seed)
    gbs = GradientBoostingSurvivalAnalysis(n_estimators=n_estimators, max_depth=5, dropout_rate=0.2, random_state=seed)
    # msa = MinlipSurvivalAnalysis()
    cgb = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=n_estimators, dropout_rate=0.2, random_state=seed)
    cox = CoxPHSurvivalAnalysis()

    estimators = {"RandomForest": rsf,
                  "GradientBoost": gbs,
                  # "MinlipSA": msa,
                  "CGBSA": cgb,
                  # "Cox": cox
                  }

    return estimators


def init_survtrace(seed):
    STConfig['data'] = 'idpp'

    STConfig["seed"] = seed


    # get model
    model = SurvTraceSingle(STConfig)

    return model

def fit_surv_trace(model, X_train, y_train, X_val, y_val):
    hparams = {
        'batch_size': 64,
        'weight_decay': 1e-4,
        'learning_rate': 1e-3,
        'epochs': 20,
    }
    # initialize a trainer
    trainer = Trainer(model)
    train_loss, val_loss = trainer.fit((X_train, y_train), (X_val, y_val),
                                       batch_size=hparams['batch_size'],
                                       epochs=hparams['epochs'],
                                       learning_rate=hparams['learning_rate'],
                                       weight_decay=hparams['weight_decay'], )
    return train_loss, val_loss

def eval_survtrace(model, df, X_train, X_val, y_val):
    evaluator = Evaluator(df, X_train.index)
    result = evaluator.eval(model, (X_val, y_val))
    print(result)
