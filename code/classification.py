# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import lightgbm


def init_classifiers():
    log_reg = LogisticRegression(max_iter=3000)
    svc = SVC()
    knn = KNeighborsClassifier(n_neighbors=5)
    gauss = GaussianNB()
    perceptron = Perceptron()
    linear_svc = LinearSVC()
    sgd = SGDClassifier()
    decision_tree = DecisionTreeClassifier()
    random_forest = RandomForestClassifier(n_estimators=200, n_jobs=-1)

    xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                      colsample_bynode=1, colsample_bytree=1, eval_metric='mlogloss',
                      gamma=0, gpu_id=-1, importance_type='gain',
                      interaction_constraints='', learning_rate=0.300000012,
                      max_delta_step=0, max_depth=10, min_child_weight=1,
                      monotone_constraints='()', n_estimators=1000, n_jobs=20,
                      num_parallel_tree=1, objective='binary:logistic', random_state=0,
                      reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,
                      tree_method='exact',
                      validate_parameters=1, verbosity=None)

    lgm = lightgbm.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                    importance_type='split', learning_rate=0.1, max_depth=-1,
                    min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                    n_estimators=1000, n_jobs=-1, num_leaves=31, objective=None,
                    random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
                    subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    models = {"LogisticRegression": log_reg,
              "SVC": svc,
              "KNN": knn,
              "GaussianNB": gauss,
              "Perceptron": perceptron,
              "LinearSVC": linear_svc,
              "SGD": sgd,
              "DecisionTree": decision_tree,
              "RandomForest": random_forest,
              "XGB": xgb,
              "LGB": lgm
              }
    return models


def fit_models(models, X, y):
    for model_name, model in models.items():
        model.fit(X, y)
    return models
