"""
Ttimeseries extraction and confound removal.

Save the output to a tar file.
"""
import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np

import nibabel as nb

from nilearn.maskers import NiftiMapsMasker, NiftiMasker
from nilearn.connectome import ConnectivityMeasure

from bids import BIDSLayout
from nilearn.interfaces.fmriprep import load_confounds_strategy


dataset_name = "ds000228"
FMRIPREP_PATH = "data/raw/fmriprep/"
BIDS_INFO =  "/lustre07/scratch/hwang1/ds000228_bids_info"
MEMORY_CACHE = "/lustre07/scratch/hwang1/nilearn_cache"

ATLAS_METADATA = {
    "path": "data/raw/segmented_difumo_atlases/tpl-MNI152NLin2009cAsym/",
    "pattern": "tpl-MNI152NLin2009cAsym_res-{resolution:02d}_atlas-DiFuMo_desc-{dimension}dimensionsSegmented_probseg",
    "dimension": [64, 128, 256, 512, 1024]
    }



def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="", epilog="""
    TBC
    """)

    parser.add_argument(
        "-s", "--subject", required=True, help="BIDS subject ID (without `sub-`)",
    )

    parser.add_argument(
        "-d", "--dim", required=False, default=-1, help=""
        "Number of dimensions in the dictionary. Valid resolutions available are {64, 128, 256, 512, 1024}, -1 for all (default: -1)",
    )

    parser.add_argument(
        "-r", "--res", required=False, default=2, help=""
        "The resolution in mm of the atlas to fetch. Valid options available are {2, 3}",
    )

    return parser

def create_timeseries_root_dir(file_entitiles):
    """Create root directory for the timeseries file."""
    subject = f"sub-{file_entitiles['subject']}"
    session = f"ses-{file_entitiles['session']}" if file_entitiles.get(
        'session', False) is not False else None
    if session:
        timeseries_root_dir = output_dir / subject / session
    else:
        timeseries_root_dir = output_dir / subject
    timeseries_root_dir.mkdir(parents=True, exist_ok=True)

    return timeseries_root_dir


def bidsish_timeseries_file_name(file_entitiles, layout, atlas_name, atlas_desc):
    """Create a BIDS-like file name to save extracted timeseries as tsv."""
    pattern = "sub-{subject}[_ses-{session}]_task-{task}[_acq-{acquisition}][_rec-{reconstruction}][_run-{run}][_echo-{echo}]"
    base = layout.build_path(file_entitiles, pattern, validate=False)
    base += f"_atlas-{atlas_name}{atlas_desc}_desc-deconfounds_timeseries.tsv"
    return base.split('/')[-1]


def get_atlas_path(dimensions, resolution):
    """Construct path of segmented difumo atlas."""
    atlas_full_path = {}

    for dimension in dimensions:
        filename = ATLAS_METADATA["pattern"].format(dimension=dimension,
                                                    resolution=resolution)
        atlas_path = os.path.join(ATLAS_METADATA["path"], filename) + ".nii.gz"
        atlas_label = atlas_path.replace("nii.gz", "tsv")
        atlas_full_path[dimension] = {"map": atlas_path, "label": atlas_label}
    return atlas_full_path


if __name__ == '__main__':
    if Path(BIDS_INFO).exists():
        print("Load existing BIDS layout.")
        layout = BIDSLayout.load(BIDS_INFO)
    else:
        print("Load create BIDS layout.")
        layout = BIDSLayout(FMRIPREP_PATH, config=['bids','derivatives'])
        layout.save(BIDS_INFO)

    dataset_name = f"dataset-{dataset_name}_timeseries"
    output_dir = Path.home() / "scratch" / dataset_name

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    args = get_parser().parse_args()
    subject = args.subject
    dimensions = ATLAS_METADATA["dimension"]
    resolution = int(args.res)
    if not (int(args.dim) == -1):
        dimensions = [int(args.dim)]
    print(args)

    atlas_full_path = get_atlas_path(dimensions, resolution)
    print(f"================ sub-{subject} ================")
    fmri = layout.get(return_type='type',
                        subject=subject,
                        space='MNI152NLin2009cAsym',
                        desc='preproc', suffix='bold', extension='nii.gz')
    file_entitiles = fmri[0].entities
    fmri = fmri[0].path
    brain_mask = layout.get(return_type='type',
                            subject=subject,
                            space='MNI152NLin2009cAsym',
                            desc='brain', suffix='mask', extension='nii.gz')
    brain_mask = brain_mask[0].path
    confounds = layout.get(return_type='file',
                            subject=subject,
                            desc='confounds', suffix='timeseries', extension='tsv')


    timeseries_root_dir = create_timeseries_root_dir(
        file_entitiles)
    confounds, sample_mask = load_confounds_strategy(fmri,
                                                        denoise_strategy='simple')

    print("time series extration...")
    for key, atlas in atlas_full_path.items():
        print("-- {} --".format(key))
        atlas_name = atlas["map"].split("atlas-")[-1].split("_")[0]
        atlas_desc = atlas["map"].split("desc-")[-1].split("_")[0]
        output_filename = bidsish_timeseries_file_name(
            file_entitiles, layout, atlas_name, atlas_desc)

        labels = pd.read_csv(atlas["label"], sep='\t')
        region_index = range(1, labels.shape[0] + 1)
        print("raw timeseries")
        masker = NiftiMapsMasker(maps_img=atlas["map"],
                                    mask_img=brain_mask,
                                    standardize=False, detrend=False,
                                    memory=MEMORY_CACHE,
                                    memory_level=1)

        raw_timeseries = masker.fit_transform(fmri)
        raw_timeseries = pd.DataFrame(raw_timeseries, columns=region_index)
        raw_timeseries.to_csv(
            timeseries_root_dir / output_filename.replace("deconfounds", "raw"),
            sep='\t', index=False)
        del raw_timeseries

        print("cleaned timeseries")
        timeseries = masker.fit_transform(fmri, confounds, sample_mask)

        # Estimating connectomes
        corr_measure = ConnectivityMeasure(kind="correlation")
        connectome = corr_measure.fit_transform([timeseries])[0]

        # save files
        timeseries = pd.DataFrame(timeseries, columns=region_index)
        timeseries.to_csv(
            timeseries_root_dir / output_filename,
            sep='\t', index=False)
        connectome = pd.DataFrame(connectome, columns=region_index, index=region_index)
        connectome.to_csv(
            timeseries_root_dir / output_filename.replace("timeseries", "connectome"),
            sep='\t')
        del connectome
        del timeseries
    print(f"=========== Finished: sub-{subject} ===========")
