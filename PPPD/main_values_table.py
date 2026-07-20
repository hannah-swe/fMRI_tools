import os
import pandas as pd
import numpy as np
from PPPD import (get_main_values_tables_path, get_selected_subject_list, get_posturography_path)
from PPPD.subjects import subs, subjects_to_exclude


part = None # needs to be None here to get th full table!
output_path = os.path.join(get_main_values_tables_path(), "full_dataframe.csv")

# --- Load main values table of both parts
main_values_tables_path = get_main_values_tables_path()
table_path_1 = os.path.join(main_values_tables_path, "PPPD_Part1_main_values_Questionnaires.xlsx")
table_path_2 = os.path.join(main_values_tables_path, "PPPD_Part2_main_values_Questionnaires.xlsx")
neo_path_1 = os.path.join(main_values_tables_path, "NEO-FFI-Auswertung.xlsx")

df1 = pd.read_excel(table_path_1, sheet_name="PPPD_values")
df2 = pd.read_excel(table_path_2, sheet_name="main_values")

ehq1_df = pd.read_excel(table_path_1, sheet_name="EHQ")
ehq2_df = pd.read_excel(table_path_2, sheet_name="EHQ")

alq1_df = pd.read_excel(table_path_1, sheet_name="ALQ_subtype")
alq2_df = pd.read_excel(table_path_2, sheet_name="ALQ_subtype")

niigata1_df = pd.read_excel(table_path_1, sheet_name="Niigata")
niigata2_df = pd.read_excel(table_path_2, sheet_name="Niigata")

neo1_df = pd.read_excel(neo_path_1, sheet_name="Tabelle1")
neo2_df = pd.read_excel(table_path_2, sheet_name="NEO_Auswertung")


# --- Choose subjects depending on experimental part and exclude subjects who participated in both parts:
selected_subs = get_selected_subject_list(part, subs, subjects_to_exclude)


# --- Change subjID in neo1_df
neo1_df["SubjID"] = neo1_df["SubjID"].str[-2:].astype(int)
neo1_df = neo1_df.rename(columns={
    "NEO.Skala_n": "Neo.Skala_n",
    "NEO.Skala_e": "Neo.Skala_e",
    "NEO.Skala_o": "Neo.Skala_o",
    "NEO.Skala_v": "Neo.Skala_v",
    "NEO.Skala_g": "Neo.Skala_g",
})


# --- Keep only selected subjects
df1 = df1[df1["SubjID"].isin(selected_subs)]
df2 = df2[df2["SubjID"].isin(selected_subs)]
ehq1_df = ehq1_df[ehq1_df["SubjID"].isin(selected_subs)]
ehq2_df = ehq2_df[ehq2_df["SubjID"].isin(selected_subs)]
alq1_df = alq1_df[alq1_df["SubjID"].isin(selected_subs)]
alq2_df = alq2_df[alq2_df["SubjID"].isin(selected_subs)]
niigata1_df = niigata1_df[niigata1_df["SubjID"].isin(selected_subs)]
niigata2_df = niigata2_df[niigata2_df["SubjID"].isin(selected_subs)]
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
    "EHQ (handedness)"
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
    "EHQ (handedness)": "EHQ",
})
# create group labels
group_values = df["Group"].map({
    3: "control",
    1: "patient",
    2: "patient",
})
# get position of "Group"
group_idx = df.columns.get_loc("Group")
# insert new column directly after "Group"
df.insert(group_idx + 1, "group", group_values)


# --- Select columns to keep for alq visual subtype df
columns_to_keep_alq = [
    "SubjID",
    "Summe Bewegung"
]
alq1_df = alq1_df[columns_to_keep_alq]
alq2_df = alq2_df[columns_to_keep_alq]
alq1_df = alq1_df.rename(columns={
    "SubjID": "subject_num",
    "Summe Bewegung": "ALQ_vis"
})
alq2_df = alq2_df.rename(columns={
    "SubjID": "subject_num",
    "Summe Bewegung": "ALQ_vis"
})

# --- Concatenate alq df
alq_df = pd.concat([alq1_df, alq2_df], ignore_index=True)

# --- Merge main values and alq df
df = df.merge(alq_df, on="subject_num", how="left")
df = df.replace(999, np.nan)


# --- Select columns to keep for alq visual subtype df
columns_to_keep_niigata = [
    "SubjID",
    "Score_3"
]
niigata1_df = niigata1_df[columns_to_keep_niigata]
niigata2_df = niigata2_df[columns_to_keep_niigata]
niigata1_df = niigata1_df.rename(columns={
    "SubjID": "subject_num",
    "Score_3": "Niigata_vis"
})
niigata2_df = niigata2_df.rename(columns={
    "SubjID": "subject_num",
    "Score_3": "Niigata_vis"
})

# --- Concatenate alq df
niigata_df = pd.concat([niigata1_df, niigata2_df], ignore_index=True)

# --- Merge main values and alq df
df = df.merge(niigata_df, on="subject_num", how="left")
df = df.replace(999, np.nan)


# --- Select columns to keep for ehq df
columns_to_keep_ehq = [
    "SubjID",
    "Händigkeit"
]
ehq1_df = ehq1_df[columns_to_keep_ehq]
ehq2_df = ehq2_df[columns_to_keep_ehq]
ehq1_df = ehq1_df.rename(columns={"SubjID": "subject_num"})
ehq2_df = ehq2_df.rename(columns={"SubjID": "subject_num"})


# --- Concatenate ehq df
ehq_df = pd.concat([ehq1_df, ehq2_df], ignore_index=True)


# --- Merge main values and ehq df
df = df.merge(ehq_df, on="subject_num", how="left")
df = df.replace(999, np.nan)


# --- Select columns to keep for neo df
columns_to_keep_neo = [
    "SubjID",
    "Neo.Skala_n",
    "Neo.Skala_e",
    "Neo.Skala_o",
    "Neo.Skala_v",
    "Neo.Skala_g",
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


# --- Load posturography data
posturography_path = get_posturography_path()
postu_path1 = os.path.join(posturography_path, "PosturoData_complete_Jul2026.xlsx")
postu_path2 = os.path.join(posturography_path, "BehavioralData_2026Jul17.xlsx")

postu1_df = pd.read_excel(postu_path1)
postu2_df = pd.read_excel(postu_path2, sheet_name="BehavioralData")


# --- Keep only selected subjects
postu1_df = postu1_df[postu1_df["VPNr"].isin(selected_subs)]
postu2_df = postu2_df[postu2_df["SubjID"].isin(selected_subs)]


# --- Select columns to keep for posturography data
columns_to_keep_postu1 = ["VPNr", "SwaySpeed.1.0.0", "SwaySpeed.0.0.0", "RatingSway.1.0.0", "RatingSway.0.0.0"]
columns_to_keep_postu2 = ["SubjID", "EOfirm", "ECfirm", "RatingEOfirm", "RatingECfirm"]

postu1_df = postu1_df[columns_to_keep_postu1]
postu2_df = postu2_df[columns_to_keep_postu2]

postu1_df = postu1_df.rename(columns={
    "VPNr": "subject_num",
    "SwaySpeed.1.0.0": "EOfirm_speed",
    "SwaySpeed.0.0.0": "ECfirm_speed",
    "RatingSway.1.0.0": "EOfirm_rating",
    "RatingSway.0.0.0": "ECfirm_rating",
})
postu2_df = postu2_df.rename(columns={
    "SubjID": "subject_num",
    "EOfirm": "EOfirm_speed",
    "ECfirm": "ECfirm_speed",
    "RatingEOfirm": "EOfirm_rating",
    "RatingECfirm": "ECfirm_rating",
})


# --- Concatenate posturography dataframes
postu_df = pd.concat([postu1_df, postu2_df], ignore_index=True)


# --- Merge main values and posturography df
df = df.merge(postu_df, on="subject_num", how="left")


# --- Save Dataframe as csv
df.to_csv(output_path, index=False)