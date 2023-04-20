import pandas as pd


def group_ts_df_by_id(orig_df, id_feat, time_series_feats, one_occurrence_feats, sort_by_this_feat):
    def group_ts_column_to_list(orig_df, id_feat, other_feature):
        unique_ids = orig_df[id_feat].unique()
        grouped_ts_column_as_list = orig_df.groupby(id_feat)[other_feature].apply(lambda x: x.values).values.tolist()
        new_df = pd.DataFrame.from_dict({id_feat: unique_ids,
                                         other_feature: grouped_ts_column_as_list})
        new_df = new_df.set_index(id_feat)
        return new_df

    def group_ts_column_select_first(orig_df, id_feat, other_feature):
        unique_ids = orig_df[id_feat].unique()
        first_values = orig_df.groupby(id_feat)[other_feature].apply(lambda x: x.values[0]).values

        new_df = pd.DataFrame.from_dict({id_feat: unique_ids,
                                         other_feature: first_values})
        new_df = new_df.set_index(id_feat)
        return new_df

    orig_df = orig_df.sort_values(by=[id_feat, sort_by_this_feat])
    ts_dfs = [group_ts_column_to_list(orig_df, id_feat, ts_feat) for ts_feat in time_series_feats]
    oo_dfs = [group_ts_column_select_first(orig_df, id_feat, oo_feat) for oo_feat in one_occurrence_feats]

    out_df = pd.concat([*oo_dfs, *ts_dfs], axis=1)
    out_df.reset_index(names=id_feat, inplace=True)
    return out_df


def transpose_df_by_uniques(orig_df, id_feat, time_series_feats, one_occurrence_feats, sort_by_this_feat):
    def transpose_cols_to_rows_by_uniques(orig_df, id_feat, other_feature):
        transposed_df = pd.DataFrame(orig_df.groupby(id_feat)[other_feature].apply(lambda x: x.values).values.tolist(),
                                     index=orig_df[id_feat].unique())
        transposed_df.columns = [f'{other_feature}_{i:02d}' for i in range(1, len(transposed_df.columns) + 1)]
        return transposed_df

    def transpose_cols_to_rows_by_1st_unique(orig_df, id_feat, other_feature):
        transposed_df = pd.DataFrame(orig_df.groupby(id_feat)[other_feature].apply(lambda x: x.values[0]),
                                     # TODO add option to store all uniques
                                     index=orig_df[id_feat].unique())
        return transposed_df

    orig_df = orig_df.sort_values(by=[id_feat, sort_by_this_feat])
    ts_dfs = [transpose_cols_to_rows_by_uniques(orig_df, id_feat, ts_feat) for ts_feat in time_series_feats]
    oo_dfs = [transpose_cols_to_rows_by_1st_unique(orig_df, id_feat, oo_feat) for oo_feat in one_occurrence_feats]
    out_df = pd.concat([*oo_dfs, *ts_dfs], axis=1)
    out_df.reset_index(names=id_feat, inplace=True)
    return out_df
