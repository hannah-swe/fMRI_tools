from pathlib import Path


# ---------------------------------------------------------------------
# HALFpipe derivatives
# ---------------------------------------------------------------------

HALFPIPE_DIR_ALL = Path(
    "/data_wgs04/ag-sensomotorik/PPPD/"
    "motion_scrubbing_check/all_subjects/derivatives/halfpipe"
)

HALFPIPE_DIR_MISSING8 = Path(
    "/data_wgs04/ag-sensomotorik/PPPD/"
    "motion_scrubbing_check/missing_8/derivatives/halfpipe"
)

# SeedFC and fALFF are now both stored in the same HALFpipe derivatives
HALFPIPE_DIRS = (
    HALFPIPE_DIR_ALL,
    HALFPIPE_DIR_MISSING8,
)


# ---------------------------------------------------------------------
# fMRIPrep derivatives
# ---------------------------------------------------------------------

FMRIPREP_DIR_ALL = Path(
    "/data_wgs04/ag-sensomotorik/PPPD/"
    "motion_scrubbing_check/all_subjects/derivatives/fmriprep"
)

FMRIPREP_DIR_MISSING8 = Path(
    "/data_wgs04/ag-sensomotorik/PPPD/"
    "motion_scrubbing_check/missing_8/derivatives/fmriprep"
)


# ---------------------------------------------------------------------
# participants.tsv
# ---------------------------------------------------------------------

PARTICIPANTS_PATH_ALL = Path(
    "/data_wgs04/ag-sensomotorik/PPPD/data/all_subjects1/participants.tsv"
)

PARTICIPANTS_PATH_MISSING8 = Path(
    "/data_wgs04/ag-sensomotorik/PPPD/data/missing8/participants.tsv"
)

PARTICIPANTS_PATHS = (
    PARTICIPANTS_PATH_ALL,
    PARTICIPANTS_PATH_MISSING8,
)


# ---------------------------------------------------------------------
# HALFpipe reportvals
# ---------------------------------------------------------------------

REPORTVALS_PATH_ALL = Path(
    "/data_wgs04/ag-sensomotorik/PPPD/"
    "motion_scrubbing_check/all_subjects/reports/reportvals.txt"
)

REPORTVALS_PATH_MISSING8 = Path(
    "/data_wgs04/ag-sensomotorik/PPPD/"
    "motion_scrubbing_check/missing_8/reports/reportvals.txt"
)

REPORTVALS_PATHS = (
    REPORTVALS_PATH_ALL,
    REPORTVALS_PATH_MISSING8,
)


# ---------------------------------------------------------------------
# Subject-dependent directories
# ---------------------------------------------------------------------

def subject_number(subject: str) -> int:
    return int(subject.removeprefix("sub-"))


def uses_missing8_dataset(subject: str) -> bool:
    return subject_number(subject) >= 173


def get_halfpipe_dir(subject: str) -> Path:
    return (
        HALFPIPE_DIR_MISSING8
        if uses_missing8_dataset(subject)
        else HALFPIPE_DIR_ALL
    )


def get_fmriprep_dir(subject: str) -> Path:
    return (
        FMRIPREP_DIR_MISSING8
        if uses_missing8_dataset(subject)
        else FMRIPREP_DIR_ALL
    )


# ---------------------------------------------------------------------
# Subjects to exclude
# ---------------------------------------------------------------------

subs_to_drop = [
    "sub-103",
    "sub-104",
    "sub-105",
    "sub-006",
    "sub-017",
    "sub-113",
    "sub-051",
    "sub-124",
    "sub-125",
    "sub-009",
    "sub-065",
    "sub-180",
]


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

out_dir = Path(
    "/data_wgs04/ag-sensomotorik/PPPD/motion_scrubbing_check/quality_check_plots")