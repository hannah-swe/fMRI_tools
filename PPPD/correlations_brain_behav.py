# TODO: 1. script with function to extract connectivity values per subject between each seed with significant cluster (OP <-> Cerebellum, V <-> Cerebellum)
#  2. get correlation_df with subject values for MMSQ, ALQ, Posturografie (eyes open, firm), Schwelle (???), Depression,
#  Anxiety, Neuroticism, duration of disease, age
#  3. get correlation matrix with all values
#  4. change connectivity df in wide format to save it as a big csv with all questionnaire and behav data

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import nibabel as nib
from PPPD import (_get_data_path, _get_derivatives_path, _get_participants_tsv, _get_full_filename, _get_output_path,
                  _get_selected_subject_list, _get_posthoc_cluster_mask, _get_signed_posthoc_map,
                  _get_cluster_table_with_aal_labels, get_main_values_tables_path)
from PPPD.extract_connectivity_values import get_dataframe_for_connectivity_values
from PPPD.subjects import subs, subjects_to_exclude
import numpy as np
import pandas as pd
import seaborn as sns

# --- Script configuration:
task = "rest"
run = "run-01" # "run-01" == pre, "run-02" == post
part = None # supported: None, 1, 2 (None: all subjects; part 1: subjects < 100; part 2: subjects >= 100)
feature = "seed_based" # supported features: "falff", "seed_based", "alff"
seeds = ["InsulaOP3RAnat", "IPLPFcmL"] # List of supported seeds:
                                    # "InsulaId1L", "InsulaId1R", "InsulaIg1L", "InsulaIg1R", "InsulaIg2L", "InsulaIg2R",
                                    # "InsulaOP3RAnat", "InsulaOP3Sphere",
                                    # "IPLPFcmL", "IPLPFcmR", "IPLPFL", "IPLPFR",
                                    # "OperculumOP1L", "OperculumOP1R", "OperculumOP2L", "OperculumOP2R", "OperculumOP4L", "OperculumOP4R",
                                    # "Precuneus",
                                    # "CSv", "CSvR",
                                    # "V1L", "V1R", "V2L", "V2R", "V5L", "V5R", "V6L", "V6R",
                                    # "VermisUvulaL", "VermisVII"
                        # for feature = "falff" or "alff" do seed = None
group_comparison = "pat>HC" # supported comparisons: "pat>HC", "HC>pat"
pre_post_diff = False
direction = "negative" # possible directions for pre-post differences:
                        # "positive" (= clusters, where pat>HC), "negative" (= cluster, where HC>pat)


# --- Load participants.tsv
participants_df = _get_participants_tsv()
participants_df["subject_id"] = participants_df["participant_id"].apply(lambda x: f"sub-{x:03d}")


# --- Choose subjects depending on experimental part and exclude subjects who participated in both parts:
selected_subs = _get_selected_subject_list(part, subs, subjects_to_exclude)


# --- Get connectivity dataframe with subject values for all significant cluster per seed
connectivity_df = get_dataframe_for_connectivity_values(seeds, feature, group_comparison, pre_post_diff, selected_subs,
                                                        participants_df, task, run, part, direction)


# --- Load full main values table for questionnaire, behavioral and posturography data
main_df_path = os.path.join(get_main_values_tables_path(), "full_dataframe.csv")
main_df = pd.read_csv(main_df_path)