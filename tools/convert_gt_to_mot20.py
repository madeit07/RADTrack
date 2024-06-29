# This script can be used to convert RADTrack GT files to track prediction output files.
# This is done for evaluating the performance of the auto-labeling solution.
# Same as convert_gt_to_pred but for single files only

from argparse import ArgumentParser, Namespace
import contextlib
import csv
import json
from pathlib import Path

import joblib
import tqdm


# https://stackoverflow.com/a/58936697
@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm.tqdm):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def get_bb(bbox3d: list[float], radar_type: str) -> tuple[float, float, float, float]:
    """Converts the 3D bounding box in RAD format to 2D RD or RA bounding box in format x_left, y_top, w, h

    Args:
        bbox3d (list[float]): 3D bounding box in RAD format
        radar_type (str): Radar format to convert to.

    Raises:
        ValueError: Radar format not supported.

    Returns:
        tuple[float, float, float, float]: 2D RD or RA bounding box in format x_left, y_top, w, h
    """
    if radar_type == 'rd':
        yc, xc, h, w = bbox3d[0], bbox3d[2], bbox3d[3], bbox3d[5]
    elif radar_type == 'ra':
        yc, xc, h, w = bbox3d[0], bbox3d[1], bbox3d[3], bbox3d[4]
    else:
        raise ValueError(f'Radar type \"{radar_type}\" not supported.')

    return (xc - w / 2), (yc - h / 2), w, h


def load_gt(file: Path) -> list[dict[str]]:
    if not file.exists():
        raise FileNotFoundError(file)

    with open(file, 'r') as f:
        return json.load(f)


def convert_gt(file: Path, args: Namespace):
    radar_type = args.radar_type.lower()
    csv_file = file.with_suffix('.txt')

    # Convert GT to MOT20 GT and Det
    gts = load_gt(file)

    with open(csv_file, 'w', newline='') as gt_csv:
        gt_writer = csv.writer(gt_csv)

        for t, gt in enumerate(gts, start=1):
            for box, id in zip(gt['boxes'], gt['ids']):
                if id < 0:
                    continue

                x, y, w, h = get_bb(box, radar_type)

                # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <consider_entry>, <class>, <visibility>
                gt_writer.writerow([t, id, x, y, w, h, 1, 1, 1])


def main(args: Namespace):
    with tqdm_joblib(tqdm.tqdm(desc='Converting GT files', total=len(args.gt_files))) as pbar:
        with joblib.Parallel(n_jobs=args.workers) as parallel:
            parallel(joblib.delayed(convert_gt)(file, args) for file in args.gt_files)


if __name__ == '__main__':
    parser = ArgumentParser(description='Script to convert the RADTrack dataset to the MOT20 format.')
    parser.add_argument('--version', '-v', action='version', version='1.0.0')
    parser.add_argument('--workers', dest='workers', type=int, default=-1, help='Number of workers for parallel execution. -1 for all CPUs.')
    parser.add_argument('--input', '-i', dest='gt_files', type=Path, required=True, nargs='+', help='Paths to the RADTrack JSON GT files to convert to MOT20 format.')
    parser.add_argument('--type', '-t', dest='radar_type', type=str, choices=['RD', 'RA'], default='RD', help='The output radar format. RD for Range-Doppler images and GT or RA for Range-Azimuth.')

    args = parser.parse_args()

    main(args)

# python convert_gt_to_mot20.py -i ..\data\autolabeling_stereo\RADTrack\radtrack-val\radtrack0005\gt\gt.json

# python convert_gt_to_mot20.py -i (Get-ChildItem -Path ..\data\autolabeling_stereo\RADTrack\ -Recurse -Filter gt.json).FullName
