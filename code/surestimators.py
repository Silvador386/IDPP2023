
from sksurv.ensemble import RandomSurvivalForest, ComponentwiseGradientBoostingSurvivalAnalysis, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastKernelSurvivalSVM, HingeLossSurvivalSVM, MinlipSurvivalAnalysis


def init_surv_estimators():
    rsf = RandomSurvivalForest(n_estimators=500, min_samples_leaf=7, random_state=0)
    gbs = GradientBoostingSurvivalAnalysis()
    # msa = MinlipSurvivalAnalysis()
    cgb = ComponentwiseGradientBoostingSurvivalAnalysis()

    estimators = {"RandomForest": rsf,
                  "GradientBoost": gbs,
                  # "MinlipSA": msa,
                  "CGBSA": cgb}

    return estimators
