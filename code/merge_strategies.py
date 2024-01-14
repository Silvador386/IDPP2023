import pandas as pd


def group_time_series_df_by_id(
        orig_df: pd.DataFrame,
        id_feature: str,
        time_series_features: list[str],
        one_occurrence_features: list[str],
        sort_by_this_feature: str
) -> pd.DataFrame:

    def group_ts_column_to_list(
            orig_df: pd.DataFrame,
            id_feature: str,
            other_feature: str
    ) -> pd.DataFrame:
        unique_ids = orig_df[id_feature].unique()
        grouped_ts_column_as_list = orig_df.groupby(id_feature)[other_feature].apply(lambda x: x.values).values.tolist()
        new_df = pd.DataFrame.from_dict({id_feature: unique_ids,
                                         other_feature: grouped_ts_column_as_list})
        new_df = new_df.set_index(id_feature)
        return new_df

    def group_ts_column_select_first(
            orig_df: pd.DataFrame,
            id_feature: str,
            other_feature: str
    ) -> pd.DataFrame:
        unique_ids = orig_df[id_feature].unique()
        first_values = orig_df.groupby(id_feature)[other_feature].apply(lambda x: x.values[0]).values

        new_df = pd.DataFrame.from_dict({id_feature: unique_ids,
                                         other_feature: first_values})
        new_df = new_df.set_index(id_feature)
        return new_df

    orig_df = orig_df.sort_values(by=[id_feature, sort_by_this_feature])
    ts_dfs = [group_ts_column_to_list(orig_df, id_feature, ts_feat) for ts_feat in time_series_features]
    oo_dfs = [group_ts_column_select_first(orig_df, id_feature, oo_feat) for oo_feat in one_occurrence_features]

    out_df = pd.concat([*oo_dfs, *ts_dfs], axis=1)
    out_df.reset_index(names=id_feature, inplace=True)
    return out_df


def transpose_df_by_uniques(
        orig_df: pd.DataFrame,
        id_feature: str,
        time_series_features: list[str],
        one_occurrence_features: list[str],
        sort_by_this_feature: str
) -> pd.DataFrame:
    def transpose_cols_to_rows_by_uniques(
            orig_df: pd.DataFrame,
            id_feature: str,
            other_feature: str
    ) -> pd.DataFrame:
        transposed_df = pd.DataFrame(orig_df.groupby(id_feature)[other_feature].apply(lambda x: x.values).values.tolist(),
                                     index=orig_df[id_feature].unique())
        transposed_df.columns = [f'{other_feature}_{i:02d}' for i in range(1, len(transposed_df.columns) + 1)]
        return transposed_df

    def transpose_cols_to_rows_by_1st_unique(
            orig_df: pd.DataFrame,
            id_feature: str,
            other_feature: str
    ) -> pd.DataFrame:
        transposed_df = pd.DataFrame(orig_df.groupby(id_feature)[other_feature].apply(lambda x: x.values[0]),
                                     # TODO add option to store all uniques
                                     index=orig_df[id_feature].unique())
        return transposed_df

    orig_df = orig_df.sort_values(by=[id_feature, sort_by_this_feature])
    ts_dfs = [transpose_cols_to_rows_by_uniques(orig_df, id_feature, ts_feat) for ts_feat in time_series_features]
    oo_dfs = [transpose_cols_to_rows_by_1st_unique(orig_df, id_feature, oo_feat) for oo_feat in one_occurrence_features]
    out_df = pd.concat([*oo_dfs, *ts_dfs], axis=1)
    out_df.reset_index(names=id_feature, inplace=True)
    return out_df
