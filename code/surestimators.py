
from sksurv.ensemble import RandomSurvivalForest, ComponentwiseGradientBoostingSurvivalAnalysis, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastKernelSurvivalSVM, HingeLossSurvivalSVM, MinlipSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis

def init_surv_estimators():
    rsf = RandomSurvivalForest(n_estimators=500, min_samples_leaf=7, random_state=0)
    gbs = GradientBoostingSurvivalAnalysis()
    # msa = MinlipSurvivalAnalysis()
    cgb = ComponentwiseGradientBoostingSurvivalAnalysis()
    cox = CoxPHSurvivalAnalysis()

    estimators = {"RandomForest": rsf,
                  "GradientBoost": gbs,
                  # "MinlipSA": msa,
                  "CGBSA": cgb,
                  # "Cox": cox
                  }

    return estimators
