from pycox.datasets import metabric, nwtco, support, gbsg, flchain
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
import pdb

from survtrace.survtrace.utils import LabelTransform
from preprocessing import fastai_ccnames


def load_data(config, merged_df, X, y_df, train_idx=None, val_idx=None):
    '''load data, return updated configuration.
    '''
    data = config['data']
    horizons = config['horizons']
    assert data in ["idpp", "metabric", "nwtco", "support", "gbsg", "flchain", "seer", ], "Data Not Found!"
    get_target = lambda df: (df['duration'].values, df['event'].values)

    if data == "idpp":
        # data processing, transform all continuous data to discrete
        cols_categorical, cols_standardize = fastai_ccnames(X)
        df = pd.concat([X, y_df], axis=1).reset_index(drop=True)
        df.rename(columns={"outcome_occurred": "event", "outcome_time": "duration"}, inplace=True)
        y_df.index = df.index
        y_df.rename(columns={"outcome_occurred": "event", "outcome_time": "duration"}, inplace=True)

        # evaluate the performance at the 25th, 50th and 75th event time quantile
        times = np.quantile(df["duration"][df["event"] == 1.0], horizons).tolist()
        # times = [2, 4, 6, 8, 10, *times]

        df_feat = df.drop(["duration", "event"], axis=1)
        df_feat_standardize = df_feat[cols_standardize]
        df_feat_standardize_disc = StandardScaler().fit_transform(df_feat_standardize)
        df_feat_standardize_disc = pd.DataFrame(df_feat_standardize_disc, columns=cols_standardize)

        # must be categorical feature ahead of numerical features!
        df_feat = pd.concat([df_feat[cols_categorical], df_feat_standardize_disc], axis=1)

        vocab_size = 0
        for _, feat in enumerate(cols_categorical):
            df_feat[feat] = df_feat[feat] + vocab_size
            vocab_size = df_feat[feat].max() + 1


        # get the largest duraiton time
        max_duration_idx = df["duration"].argmax()
        if train_idx is None and val_idx is None:
            df_test = df_feat.drop(max_duration_idx).sample(frac=0.1)
            df_train = df_feat.drop(df_test.index)
            df_val = df_train.drop(max_duration_idx).sample(frac=0.1)
            df_train = df_train.drop(df_val.index)
        else:
            df_train = df_feat.iloc[train_idx]
            df_val = df_feat.iloc[val_idx]
            # df_test = df_feat.drop(max_duration_idx).sample(frac=0.1)
            # df_train = df_train.drop(df_test.index)
            df_test = pd.DataFrame()
        # assign cuts
        labtrans = LabelTransform(cuts=np.array([df["duration"].min()] + times + [df["duration"].max()]))
        labtrans.fit(*get_target(df.loc[df_train.index]))
        y = labtrans.transform(*get_target(df))  # y_struct = (discrete duration, event indicator)
        df_y_train = pd.DataFrame(
            {"duration": y[0][df_train.index], "event": y[1][df_train.index], "proportion": y[2][df_train.index]},
            index=df_train.index)
        df_y_val = pd.DataFrame(
            {"duration": y[0][df_val.index], "event": y[1][df_val.index], "proportion": y[2][df_val.index]},
            index=df_val.index)
        # df_y_test = pd.DataFrame({"duration": y_struct[0][df_test.index], "event": y_struct[1][df_test.index],  "proportion": y_struct[2][df_test.index]}, index=df_test.index)
        if df_test.empty:
            df_y_test = pd.DataFrame()
        else:
            df_y_test = pd.DataFrame(
                {"duration": df['duration'].loc[df_test.index], "event": df['event'].loc[df_test.index]})


    config['labtrans'] = labtrans
    config['num_numerical_feature'] = int(len(cols_standardize))
    config['num_categorical_feature'] = int(len(cols_categorical))
    config['num_feature'] = int(len(df_train.columns))
    config['vocab_size'] = int(vocab_size)
    config['duration_index'] = labtrans.cuts
    config['out_feature'] = int(labtrans.out_features)
    return df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val
