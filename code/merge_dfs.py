import pandas as pd
from merge_strategies import transpose_df_by_uniques, group_ts_df_by_id


def merge_dfs(dfs, dataset_name, id_feat, dataset_type="train"):
    prep_df_for_merge = transpose_df_by_uniques

    if dataset_type == "train":
        merged_df = pd.merge(dfs[f"{dataset_name}_train-static-vars"], dfs[f"{dataset_name}_train-outcomes"],
                             on="patient_id", how="outer")
    else:
        merged_df = dfs[f"{dataset_name}_test-static-vars"]

    relapses_df = dfs[f"{dataset_name}_{dataset_type}-relapses"]
    ts_feats = ["delta_relapse_time0"]
    oo_feats = ["centre"]
    sort_by_this_feat = "delta_relapse_time0"
    relapses_df = prep_df_for_merge(relapses_df, id_feat, ts_feats, oo_feats, sort_by_this_feat)

    ms_type_df = dfs[f"{dataset_name}_{dataset_type}-ms-type"]
    ts_feats = ["multiple_sclerosis_type", "delta_observation_time0"]
    oo_feats = ["centre"]
    sort_by_this_feat = "delta_observation_time0"
    ms_type_df = prep_df_for_merge(ms_type_df, id_feat, ts_feats, oo_feats, sort_by_this_feat)

    mri_df = dfs[f"{dataset_name}_{dataset_type}-mri"]
    ts_feats = ["mri_area_label", "lesions_T1", "lesions_T1_gadolinium", "number_of_lesions_T1_gadolinium",
                "new_or_enlarged_lesions_T2", "number_of_new_or_enlarged_lesions_T2", "lesions_T2",
                "number_of_total_lesions_T2", "delta_mri_time0"
                ]
    oo_feats = ["centre"]
    sort_by_this_feat = "delta_mri_time0"
    mri_df = prep_df_for_merge(mri_df, id_feat, ts_feats, oo_feats, sort_by_this_feat)

    evoked_p_df = dfs[f"{dataset_name}_{dataset_type}-evoked-potentials"]
    ts_feats = ["altered_potential", "potential_value", "location", "delta_evoked_potential_time0"]
    oo_feats = ["centre"]
    sort_by_this_feat = "delta_evoked_potential_time0"
    evoked_p_df = prep_df_for_merge(evoked_p_df, id_feat, ts_feats, oo_feats, sort_by_this_feat)

    edss_df = dfs[f"{dataset_name}_{dataset_type}-edss"]
    ts_feats = ["edss_as_evaluated_by_clinician", "delta_edss_time0"]
    oo_feats = ["centre"]
    sort_by_this_feat = "delta_edss_time0"
    edss_df = prep_df_for_merge(edss_df, id_feat, ts_feats, oo_feats, sort_by_this_feat)

    grouped_dfs = [edss_df, relapses_df, ms_type_df, evoked_p_df, mri_df]
    for df in grouped_dfs:
        merged_df = pd.merge(merged_df, df, on=[id_feat, "centre"], how="outer")

    return merged_df


def merge_multiple(multi_merge_dfs, id_feat):
    patient_ids = [merge_df[id_feat] for merge_df in multi_merge_dfs]
    new_merged_df = pd.concat(multi_merge_dfs, axis=0, ignore_index=True)
    return patient_ids, new_merged_df
