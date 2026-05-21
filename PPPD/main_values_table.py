# TODO: add posturography values

import os
import pandas as pd
import numpy as np
from PPPD import (get_main_values_tables_path, _get_selected_subject_list)
from PPPD.subjects import subs, subjects_to_exclude

part = None # needs to be None here to get th full table!


# --- Load main values table of both parts
main_values_tables_path = get_main_values_tables_path()
table_path_1 = os.path.join(main_values_tables_path, "PPPD_Part1_main_values_Questionnaires.xlsx")
table_path_2 = os.path.join(main_values_tables_path, "PPPD_Part2_main_values_Questionnaires.xlsx")
neo_path_1 = os.path.join(main_values_tables_path, "NEO-FFI-Auswertung.xlsx")

df1 = pd.read_excel(table_path_1, sheet_name="PPPD_values")
df2 = pd.read_excel(table_path_2, sheet_name="main_values")

neo1_df = pd.read_excel(neo_path_1, sheet_name="Tabelle1")
neo2_df = pd.read_excel(table_path_2, sheet_name="NEO_Auswertung")


# --- Choose subjects depending on experimental part and exclude subjects who participated in both parts:
selected_subs = _get_selected_subject_list(part, subs, subjects_to_exclude)


# --- Change subjID in neo1_df
neo1_df["SubjID"] = neo1_df["SubjID"].str[-2:].astype(int)
neo1_df = neo1_df.rename(columns={"NEO.Skala_n": "Neo.Skala_n"})


# --- Keep only selected subjects
df1 = df1[df1["SubjID"].isin(selected_subs)]
df2 = df2[df2["SubjID"].isin(selected_subs)]
neo1_df = neo1_df[neo1_df["SubjID"].isin(selected_subs)]
neo2_df = neo2_df[neo2_df["SubjID"].isin(selected_subs)]


# --- Select columns to keep
columns_to_keep = [
    "SubjID",
    "Group",
    "Age (in years)",
    "Age2",
    "disease duration (in months)",
    "Gender",
    "GVSthresholdMRI",
    "GVSthresholdBehav",
    "ALQ_total",
    "Niigata_total",
    "HADS_A_total",
    "HADS_D_total",
    "MSSQ_raw",
]

df1 = df1[columns_to_keep]
df2 = df2[columns_to_keep]


# --- concatenate df1 and df2
assert list(df1.columns) == list(df2.columns)
df = pd.concat([df1, df2], ignore_index=True)
df = df.sort_values(by="SubjID").reset_index(drop=True)
df = df.rename(columns={
    "SubjID": "subject_num",
    "Age (in years)": "age_in_years",
    "Age2": "age",
    "disease duration (in months)": "disease_duration",
    "Gender": "gender",
    "GVSthresholdMRI": "GVS_threshold_mri",
    "GVSthresholdBehav": "GVS_threshold_behav",
})


# --- Select columns to keep for neo df
columns_to_keep_neo = [
    "SubjID",
    "Neo.Skala_n"
]
neo1_df = neo1_df[columns_to_keep_neo]
neo2_df = neo2_df[columns_to_keep_neo]
neo1_df = neo1_df.rename(columns={"SubjID": "subject_num"})
neo2_df = neo2_df.rename(columns={"SubjID": "subject_num"})


# --- Concatenate neo df
neo_df = pd.concat([neo1_df, neo2_df], ignore_index=True)


# --- Merge main values and neo df
df = df.merge(neo_df, on="subject_num", how="left")
df = df.replace(999, np.nan)