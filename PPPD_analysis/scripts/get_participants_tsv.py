import pandas as pd
from bad_subjects import exclude_part1, exclude_part2

# get .tsv file for participants of PPPD Part 1
# path to input and output files
input_file1 = "W:/PPPD/Auswertung_Part1/PPPD_main_values_Questionnaires.xlsx"
output_file1 = "W:/PPPD/Auswertung_Part1/MRI_Restingstate/participants_PPPD1.tsv"

# load excel file
df1 = pd.read_excel(input_file1)
df1["participant_id"] = df1["SubjID"].astype(str).str.zfill(3)

# exclude subjects
df1 = df1[df1["MRI-Data"] == 1]
df1 = df1[~df1["participant_id"].isin(exclude_part1)]

# get new columns
df1["age"] = df1["Age2"].astype(int)
df1["sex"] = df1["Gender"].str.upper()
df1["group"] = df1["Group"].apply(lambda x: "control" if x == 3 else "patient")
# mri_pat_id = "CBBM" + MRI_Number
df1["mri_pat_id"] = "CBBM" + df1["MRI-Number"].astype(int).astype(str)

# keep only necessary columns and set their oder
final_df1 = df1[[
    "participant_id",
    "age",
    "sex",
    "group",
    "mri_pat_id"
]]

# save as .tsv
final_df1.to_csv(output_file1, sep="\t", index=False)
print("tsv-file successfully saved:", output_file1)


# get .tsv file for participants of PPPD Part 2
# path to input and output files
input_file2 = "W:/PPPD/Auswertung_Part2/PPPD_Part2_main_values_Questionnaires.xlsx"
output_file2 = "W:/PPPD/Auswertung_Part2/MRI/RestingState/participants_PPPD2.tsv"

# load excel file
df2 = pd.read_excel(input_file2)
df2["participant_id"] = df2["SubjID"].astype(str).str.zfill(3)

# exclude subjects
df2 = df2[df2["Ausschluss"] == 0]
df2 = df2[~df2["participant_id"].isin(exclude_part2)]

# get new columns
df2["age"] = df2["Age2"].astype(int)
df2["sex"] = df2["Gender"].str.upper()
df2["group"] = df2["Group"].apply(lambda x: "control" if x == 3 else "patient")
# mri_pat_id = "CBBM" + MRI_Number
df2["mri_pat_id"] = "CBBM" + df2["MRI_Number"].astype(int).astype(str)

# keep only necessary columns and set their oder
final_df2 = df2[[
    "participant_id",
    "age",
    "sex",
    "group",
    "mri_pat_id"
]]

# save as .tsv
final_df2.to_csv(output_file2, sep="\t", index=False)
print("tsv-file successfully saved:", output_file2)

