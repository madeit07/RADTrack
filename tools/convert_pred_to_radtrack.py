# This script is used to convert tracker prediction results in MOT20 format back to RADTrack JSON files.

from collections import defaultdict
from argparse import ArgumentParser, Namespace
import contextlib
import csv
import json
from pathlib import Path

import joblib
import tqdm

# ID is index + 1 -> person = ID1, bicycle = ID2, ...
ALL_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']


class defaultlist(list):
    def __init__(self, fx):
        self._defaultconstr = fx

    def _fill(self, index):
        while len(self) <= index:
            self.append(self._defaultconstr())

    def __setitem__(self, index, value):
        self._fill(index)
        list.__setitem__(self, index, value)

    def __getitem__(self, index):
        self._fill(index)
        return list.__getitem__(self, index)


class Instance(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            dict.__setitem__(self, k, v)

    @classmethod
    def from_csv_row(self, row: list[str], ignore_class: bool = False):
        frame = int(row[0])
        id = int(row[1])
        x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
        xc, yc, = x + w / 2, y + h / 2
        include = bool(row[6])
        cls_index = int(row[7]) - 1

        if ignore_class or (cls_index >= 0 and cls_index < len(ALL_CLASSES)):
            cls = ''
        else:
            cls = ALL_CLASSES[cls_index]

        visibility = float(row[8])

        return self(frame=frame, id=id, xc=xc, yc=yc, w=w, h=h, include=include, cls=cls, visibility=visibility)


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


def get_bb(instance: Instance, radar_type: str) -> list[float]:
    if radar_type == 'rd':
        bbox3d = [instance.yc, 0, instance.xc, instance.h, 0, instance.w]
    elif radar_type == 'ra':
        bbox3d = [instance.yc, instance.xc, 0, instance.h, instance.w, 0]
    else:
        bbox3d = [0, 0, 0, 0, 0, 0]

    return bbox3d


def get_track_result_files(dir: Path):
    return list(dir.glob('*track*.txt'))


def convert(track_file: Path, output_dir: Path, args: Namespace):
    radar_type = args.radar_type.lower()
    tracks: list[dict[str, list]] = defaultlist(lambda: defaultdict(list))

    with open(track_file, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            instance = Instance.from_csv_row(row, args.ignore_class)
            frame_idx = instance.frame - 1

            if not instance.include:
                continue

            tracks[frame_idx]['boxes'].append(get_bb(instance, radar_type))
            tracks[frame_idx]['cart_boxes'].append([0, 0, 0, 0])
            tracks[frame_idx]['classes'].append(instance.cls)
            tracks[frame_idx]['ids'].append(instance.id)

    output_filename = track_file.with_suffix('.json').name
    with open(output_dir / output_filename, 'w') as f:
        json.dump(tracks, f, separators=(',', ':'))


def main(args: Namespace):
    if not args.prediction_dir.is_dir():
        raise ValueError(f'\"{args.prediction_dir}\" is not a directory.')

    if args.output_dir is None:
        args.output_dir = args.prediction_dir

    if not args.output_dir.is_dir():
        raise ValueError(f'\"{args.output_dir}\" is not a directory.')

    track_files = get_track_result_files(args.prediction_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tqdm_joblib(tqdm.tqdm(desc='Converting track files', total=len(track_files))) as pbar:
        with joblib.Parallel(n_jobs=args.workers) as parallel:
            parallel(joblib.delayed(convert)(file, args.output_dir, args) for file in track_files)


if __name__ == '__main__':
    parser = ArgumentParser(description='Script to convert the MOT20 tracking results to RADTrack format.')
    parser.add_argument('--version', '-v', action='version', version='1.0.0')
    parser.add_argument('--workers', dest='workers', type=int, default=-1, help='Number of workers for parallel execution. -1 for all CPUs.')
    parser.add_argument('--prediction', '-p', dest='prediction_dir', type=Path, required=True, help='Path to the folder containing the tracker results. Each sequence is in its own CSV file.')
    parser.add_argument('--type', '-t', dest='radar_type', type=str, choices=['RD', 'RA'], default='RD', help='The radar format the track was performed in. RD for Range-Doppler or RA for Range-Azimuth.')
    parser.add_argument('--output', '-o', dest='output_dir', type=Path, default=None, help='The output directory for the JSON files.')
    parser.add_argument('--ignore-class', '-nc', dest='ignore_class', action='store_true', help='Sets the predication classes to empty values.')

    args = parser.parse_args()

    main(args)

# python convert_pred_to_radtrack.py -p ..\data\trackers\rdtrack-val\run27_e99_ms5
