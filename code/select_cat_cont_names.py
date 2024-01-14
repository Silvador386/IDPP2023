import pandas as pd


def fastai_ccnames_original(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    col_value_types = {"bool": ['ms_in_pediatric_age', 'spinal_cord_symptom', 'brainstem_symptom', 'eye_symptom',
                                'supratentorial_symptom'],
                       "int32": ['new_or_enlarged_lesions_T2_5+', 'number_of_new_or_enlarged_lesions_T2_5+',
                                 'altered_potential_9+', 'potential_value_9+', 'delta_relapse_time0_3+',
                                 'mri_area_label_6+',
                                 'delta_mri_time0_6+', 'lesions_T1_3+', 'lesions_T2_3+',
                                 'delta_evoked_potential_time0_9+',
                                 'lesions_T1_gadolinium_5+', 'number_of_lesions_T1_gadolinium_6+',
                                 'edss_as_evaluated_by_clinician_11+',
                                 'location_9+',
                                 'delta_edss_time0_10+',
                                 'number_of_total_lesions_T2_3+'],
                       "int64": ['age_at_onset', 'time_since_onset'],
                       "float64": ['diagnostic_delay', 'delta_relapse_time0_01', 'delta_relapse_time0_02',
                                   'delta_observation_time0_01', 'delta_observation_time0_02',
                                   'number_of_lesions_T1_gadolinium_01', 'number_of_lesions_T1_gadolinium_02',
                                   'number_of_lesions_T1_gadolinium_03', 'number_of_lesions_T1_gadolinium_04',
                                   'number_of_lesions_T1_gadolinium_05', 'number_of_new_or_enlarged_lesions_T2_01',
                                   'number_of_new_or_enlarged_lesions_T2_02', 'number_of_new_or_enlarged_lesions_T2_03',
                                   'number_of_new_or_enlarged_lesions_T2_04', 'delta_mri_time0_01',
                                   'delta_mri_time0_02',
                                   'delta_mri_time0_03', 'delta_mri_time0_04', 'delta_mri_time0_05',
                                   'delta_evoked_potential_time0_01', 'delta_evoked_potential_time0_02',
                                   'delta_evoked_potential_time0_03', 'delta_evoked_potential_time0_04',
                                   'delta_evoked_potential_time0_05', 'delta_evoked_potential_time0_06',
                                   'delta_evoked_potential_time0_07', 'delta_evoked_potential_time0_08',
                                   'edss_as_evaluated_by_clinician_01',
                                   'edss_as_evaluated_by_clinician_02',
                                   'edss_as_evaluated_by_clinician_03', 'edss_as_evaluated_by_clinician_04',
                                   'edss_as_evaluated_by_clinician_05', 'edss_as_evaluated_by_clinician_06',
                                   'edss_as_evaluated_by_clinician_07', 'edss_as_evaluated_by_clinician_08',
                                   'edss_as_evaluated_by_clinician_09', 'edss_as_evaluated_by_clinician_10',
                                   'delta_edss_time0_01',
                                   'delta_edss_time0_02', 'delta_edss_time0_03',
                                   'delta_edss_time0_04', 'delta_edss_time0_05', 'delta_edss_time0_06',
                                   'delta_edss_time0_07', 'delta_edss_time0_08', 'delta_edss_time0_09'],
                       "object": ['sex', 'residence_classification', 'ethnicity', 'other_symptoms', 'centre',
                                  'multiple_sclerosis_type_01', 'multiple_sclerosis_type_02', 'mri_area_label_01',
                                  'mri_area_label_02', 'mri_area_label_03', 'mri_area_label_04', 'mri_area_label_05',
                                  'lesions_T1_01', 'lesions_T1_02', 'lesions_T1_gadolinium_01',
                                  'lesions_T1_gadolinium_02',
                                  'new_or_enlarged_lesions_T2_01', 'new_or_enlarged_lesions_T2_02',
                                  'new_or_enlarged_lesions_T2_03', 'new_or_enlarged_lesions_T2_04', 'lesions_T2_01',
                                  'lesions_T2_02', 'number_of_total_lesions_T2_01', 'number_of_total_lesions_T2_02',
                                  'altered_potential_01', 'altered_potential_02', 'altered_potential_03',
                                  'altered_potential_04', 'altered_potential_05', 'altered_potential_06',
                                  'altered_potential_07', 'altered_potential_08', 'potential_value_01',
                                  'potential_value_02',
                                  'potential_value_03', 'potential_value_04', 'potential_value_05',
                                  'potential_value_06',
                                  'potential_value_07', 'potential_value_08', 'location_01', 'location_02',
                                  'location_03',
                                  'location_04', 'location_05', 'location_06', 'location_07', 'location_08'
                                  ]
                       }
    col_value_types = df.columns.to_series().groupby(df.dtypes).groups

    col_value_types = {f"{key}": value for key, value in col_value_types.items()}

    cat_names = [*col_value_types["bool"], *col_value_types["object"]]
    cont_names = [*col_value_types["int64"], *col_value_types["float64"]]
    cont_names.remove("outcome_occurred")
    cont_names.remove("outcome_time")

    return cat_names, cont_names
