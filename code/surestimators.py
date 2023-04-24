from sksurv.ensemble import RandomSurvivalForest, ComponentwiseGradientBoostingSurvivalAnalysis, \
    GradientBoostingSurvivalAnalysis
from sksurv.svm import FastKernelSurvivalSVM, HingeLossSurvivalSVM, MinlipSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis


def init_surv_estimators(seed):
    rsf = RandomSurvivalForest(n_estimators=500, min_samples_leaf=7, random_state=seed)
    gbs = GradientBoostingSurvivalAnalysis(random_state=seed)
    # msa = MinlipSurvivalAnalysis()
    cgb = ComponentwiseGradientBoostingSurvivalAnalysis(random_state=seed)
    cox = CoxPHSurvivalAnalysis()

    estimators = {"RandomForest": rsf,
                  "GradientBoost": gbs,
                  # "MinlipSA": msa,
                  "CGBSA": cgb,
                  # "Cox": cox
                  }

    return estimators
