import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxPHSurvivalAnalysis


def clr_acc(models, X_train, y_train, X_valid, y_valid):
    accuracies = {}
    for model_name, model in models.items():
        train_acc = round(model.score(X_train, y_train) * 100, 2)
        val_acc = round(model.score(X_valid, y_valid) * 100, 2)
        accuracies[model_name] = {"train_acc": train_acc, "val_acc": val_acc}

    acc_df = pd.DataFrame(accuracies).transpose().sort_values("val_acc", ascending=False)

    return acc_df


def evaluate_regressors_rmsle(regressors, X, y):
    def rmsle_cv(model, X_train, y_train, n_folds=5):
        kf = KFold(n_folds, shuffle=True, random_state=43).get_n_splits(X_train.values)
        rmse = np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
        return (rmse)

    for model_name, model in regressors.items():
        score = rmsle_cv(model, X, y)
        print(f"{model_name} score: {score.mean():.4f} ({score.std():.4f})")


def evaluate_estimators(estimator, X, y, plot=True, print_coef=False):
    # prediction = model.predict(X)
    # result = concordance_index_censored([i for i, _ in y], [i for _, i in y], prediction)

    c_score = estimator.score(X, y)

    def fit_and_score_features(X, y):
        n_features = X.shape[1]
        scores = np.empty(n_features)
        m = CoxPHSurvivalAnalysis()
        for j in range(n_features):
            Xj = X[:, j:j + 1]
            m.fit(Xj, y)
            scores[j] = m.score(Xj, y)
        return scores

    if print_coef:
        scores = fit_and_score_features(X.values, y)
        print("Importance of features:\n", pd.Series(scores, index=X.columns).sort_values(ascending=False))

    if plot:
        plot_cox_step_funcs(estimator, X, print_coef)
        plot_kaplan([status for status, _ in y], [time for _, time in y])

    return c_score


def plot_cox_step_funcs(estimator, X, print_coefs=False):
    pred_surv = estimator.predict_survival_function(X)
    time_points = np.arange(1, 15)
    for i, surv_func in enumerate(pred_surv):
        plt.step(time_points, surv_func(time_points), where="post",
                 label="Sample %d" % (i + 1))
    plt.title("CoxPHSurvival Step functions")
    plt.ylabel("est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
    plt.ylim([0, 1])
    plt.legend(loc="best")
    plt.show()

    if print_coefs:
        print(pd.Series(estimator.coef_, index=X.columns))


def plot_kaplan(status, time):

    time, survival_prob = kaplan_meier_estimator(status, [i for _, i in time])
    plt.step(time, survival_prob, where="post")
    plt.title("Kaplan-Meier model")
    plt.ylabel("est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
    plt.ylim([0, 1])
    plt.show()


def evaluate_c_index(classifiers, regressors):
    from lifelines.utils.concordance import concordance_index
    # c_index = concordance_index(regression_preds, classification_preds)