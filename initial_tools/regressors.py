# machine learning
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from xgboost import XGBRegressor
import lightgbm


def init_regressors():
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber', random_state=5)
    model_xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213,
                             random_state=7, nthread=-1)
    model_lgb = lightgbm.LGBMRegressor(objective='regression', num_leaves=5,
                                       learning_rate=0.05, n_estimators=720,
                                       max_bin=55, bagging_fraction=0.8,
                                       bagging_freq=5, feature_fraction=0.2319,
                                       feature_fraction_seed=9, bagging_seed=9,
                                       min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

    regressors = {"Lasso": lasso, "ElasticNet": ENet, "Kernel Ridge": KRR,
                  "Gradient Boosting": GBoost, "XGBoost": model_xgb,
                  "LGBM": model_lgb
                  }
    return regressors
