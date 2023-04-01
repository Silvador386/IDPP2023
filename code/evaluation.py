import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split


def clr_acc(models, X_train, y_train, X_valid, y_valid):
    accuracies = {}
    for model_name, model in models.items():
        train_acc = round(model.score(X_train, y_train) * 100, 2)
        val_acc = round(model.score(X_valid, y_valid) * 100, 2)
        accuracies[model_name] = {"train_acc": train_acc, "val_acc": val_acc}

        if model_name == "LGB":
            classification_preds = model.predict(X_valid)
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


def evaluate_c_index(classifiers, regressors):
    pass