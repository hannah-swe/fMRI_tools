from pathlib import Path
import os
import shutil
import yaml
from subjects import subs


# --- List of all supported seeds
# all seed names that are in seed-based connectivity analysis folder 1
SEEDS_1 = [
    "InsulaId1L", "InsulaId1R", "InsulaIg1L", "InsulaIg1R", "InsulaIg2L", "InsulaIg2R",
    "InsulaOP3RAnat", "InsulaOP3Sphere",
    "IPLPFcmL", "IPLPFcmR", "IPLPFL", "IPLPFR",
    "OperculumOP1L", "OperculumOP1R", "OperculumOP2L", "OperculumOP2R",
    "OperculumOP4L", "OperculumOP4R",
    "Precuneus"
]

# all seed names that are in seed-based connectivity analysis folder 2
SEEDS_2 = [
    "CSv", "CSvR",
    "V1L", "V1R", "V2L", "V2R", "V5L", "V5R", "V6L", "V6R",
    "VermisUvulaL", "VermisVII"
]

# all seed names that are in seed-based connectivity analysis folder 3
SEEDS_3 = [
    "HippocampusL", "HippocampusR"
]

seeds = SEEDS_1 + SEEDS_2 + SEEDS_3


# --- Function to get the full filename
def get_filename(subject_id, task, run, feature, seed=None):
    if feature == "seed_based":
        return f"{subject_id}_task-{task}_{run}_feature-seedbased_seed-{seed}_stat-effect_statmap.nii.gz"
    elif feature == "falff":
        return f"{subject_id}_task-{task}_{run}_feature-fALFF_falff.nii.gz"
    elif feature == "alff":
        return f"{subject_id}_task-{task}_{run}_feature-fALFF_alff.nii.gz"
    else:
        raise ValueError(f"Unsupported feature: {feature}")


# --- Load config.yml and get analysis path
PROJECT_DIR = Path.cwd()
config_file = PROJECT_DIR / "config.yml"
with open(config_file, "r") as f:
    config = yaml.safe_load(f)
analysis_path = config["paths"]["analysis_path"]
analysis_path = str(analysis_path)


# --- Data path structure in HALFpipe working directories
data_paths = {
    "seed1": os.path.join(analysis_path, "both_parts_seed1", "derivatives", "halfpipe"),
    "seed2": os.path.join(analysis_path, "both_parts_seed2", "derivatives", "halfpipe"),
    "seed3": os.path.join(analysis_path, "both_parts_seed3", "derivatives", "halfpipe"),
    "falff": os.path.join(analysis_path, "both_parts_falff", "derivatives", "halfpipe"),
    "missing8": os.path.join(analysis_path, "missing8", "derivatives", "halfpipe"),
    "missing8_2": os.path.join(analysis_path, "missing8_2", "derivatives", "halfpipe"),
}


# --- Make new data directory for all collected and relevant HALFpipe results
new_data_dir = os.path.join(analysis_path, "HALFpipe_output")
os.makedirs(new_data_dir, exist_ok=True)


# --- Lists of task, runs and features to copy data from
task = "rest"
runs = ["run-01", "run-02"]
features = ["falff", "seed_based"]


# --- Initialize lists to store copied and missing files
missing_files = []
copied_files = []


# --- Loop over subjects, features, runs and seeds to copy relevant data
for s in subs:
    subject_id = f"sub-{s:03d}"
    # new subject folder
    subject_out_dir = os.path.join(new_data_dir, subject_id)
    os.makedirs(subject_out_dir, exist_ok=True)

    for feature in features:
        for run in runs:
            if feature == "falff":
                if s >= 173:
                    base_path = data_paths["missing8"]
                else:
                    base_path = data_paths["falff"]

                # statmap
                filename = get_filename(subject_id, task, run, feature)
                source_file = os.path.join(base_path, subject_id, "func", f"task-{task}", filename)
                target_file = os.path.join(subject_out_dir, filename)

                if os.path.exists(source_file):
                    shutil.copy2(source_file, target_file)
                    copied_files.append(source_file)
                else:
                    missing_files.append(source_file)

                # mask
                mask_filename = f"{subject_id}_task-{task}_{run}_feature-fALFF_mask.nii.gz"
                mask_source = os.path.join(base_path, subject_id, "func", f"task-{task}", mask_filename)
                mask_target = os.path.join(subject_out_dir, mask_filename)

                if os.path.exists(mask_source):
                    shutil.copy2(mask_source, mask_target)
                    copied_files.append(mask_source)
                else:
                    missing_files.append(mask_source)

            elif feature == "seed_based":
                for seed in seeds:
                    # statmap
                    filename = get_filename(subject_id, task, run, feature, seed)

                    if s >= 173:
                        if seed in SEEDS_1 and SEEDS_2:
                            base_path = data_paths["missing8"]
                        if seed in SEEDS_3:
                            base_path = data_paths["missing8_2"]
                    else:
                        if seed in SEEDS_1:
                            base_path = data_paths["seed1"]
                        elif seed in SEEDS_2:
                            base_path = data_paths["seed2"]
                        elif seed in SEEDS_3:
                            base_path = data_paths["seed3"]
                        else:
                            raise ValueError(f"Unknown seed: {seed}")

                    source_file = os.path.join(base_path, subject_id, "func", f"task-{task}", filename)
                    target_file = os.path.join(subject_out_dir, filename)

                    if os.path.exists(source_file):
                        shutil.copy2(source_file, target_file)
                        copied_files.append(source_file)
                    else:
                        missing_files.append(source_file)

                    # mask
                    mask_filename = (f"{subject_id}_task-{task}_{run}_feature-seedbased_seed-{seed}_mask.nii.gz")

                    mask_source = os.path.join(base_path, subject_id, "func", f"task-{task}", mask_filename)
                    mask_target = os.path.join(subject_out_dir, mask_filename)

                    if os.path.exists(mask_source):
                        shutil.copy2(mask_source, mask_target)
                        copied_files.append(mask_source)
                    else:
                        missing_files.append(mask_source)


print(f"Copied files: {len(copied_files)}")
print(f"Missing files: {len(missing_files)}")

if missing_files:
    print("\nMissing files:")
    for file in missing_files[:20]:
        print(file)

    if len(missing_files) > 20:
        print(f"... and {len(missing_files) - 20} more")
