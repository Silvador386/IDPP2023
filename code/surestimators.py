from sksurv.ensemble import RandomSurvivalForest, ComponentwiseGradientBoostingSurvivalAnalysis, \
    GradientBoostingSurvivalAnalysis
from sksurv.svm import FastKernelSurvivalSVM, HingeLossSurvivalSVM, MinlipSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis


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
