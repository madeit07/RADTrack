# This script is used to convert from RADTrack format to MOT20 format but only for either RD or RA case
# as the MOT20 format only support 2D bounding box tracking.

from argparse import ArgumentParser, Namespace
import configparser
import contextlib
import csv
import json
import os
import sys
from pathlib import Path
import shutil

import joblib
import tqdm
import numpy as np
import cv2

# Used for creating a junction
if sys.platform == 'win32':
    import _winapi


DATASET_NAME = 'radtrack'

def rad_to_rd(rad: np.ndarray) -> np.ndarray:
    magnitude = pow(np.abs(rad), 2)
    rd = np.sum(magnitude, axis=1)
    rd = 10 * np.log10(rd + 1.)

    return rd


def rad_to_ra(rad: np.ndarray) -> np.ndarray:
    magnitude = pow(np.abs(rad), 2)
    ra = np.sum(magnitude, axis=-1)
    ra = 10 * np.log10(ra + 1.)

    return ra

def load_gt(seq_dir: Path) -> list[dict[str]]:
    file_path = seq_dir / 'gt' / 'gt.json'
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    with open(file_path, 'r') as f:
        return json.load(f)

def load_rad(seq_dir: Path, t: int) -> np.ndarray:
    file_path = seq_dir / 'RAD' / f'{t:06d}.npy'
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    return np.load(file_path)

def norm_to_image(array: np.ndarray) -> np.ndarray:
    """Normalize to image format"""

    # Normalized [0,1]
    n = (array - np.min(array)) / np.ptp(array)

    # Normalized [0,255]
    return (255 * n).astype(np.uint8)

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


def get_sequence_paths(data_path: Path) -> tuple[list[tuple[str, str, Path]], list[str]]:
    """Returns the paths to all sequences in the dataset.

    Args:
        data_path (Path): Path to the dataset containing the splits and sequences.

    Returns:
        tuple[list[tuple[str, str, Path]], list[str]]: List of sequence names, its split and path and all splits
    """
    seqmaps_dir = data_path / 'seqmaps'
    seqmaps = [f for f in seqmaps_dir.iterdir() if f.is_file()]
    splits = [f.stem.split('-')[-1] for f in seqmaps]

    # seq_name, split, sequence_dir
    sequences: list[tuple[str, str, Path]] = []
    for split, seqmap in zip(splits, seqmaps):
        split_dir = data_path / f'{DATASET_NAME}-{split}'

        with open(seqmap, 'r', newline='') as f:
            seq_reader = csv.reader(f)
            sequences.extend([(seq[0], split, split_dir / seq[0]) for seq in seq_reader if seq[0] != 'name'])

    return sequences, splits

def parse_sequence_info(sequence_dir: Path) -> dict[str, int | str]:
    """Returns all meta information about the sequence.

    Args:
        sequence_dir (str): Path to the sequence directory.

    Raises:
        FileNotFoundError: Sequence info ini is missing.

    Returns:
        dict[str, int | str]: Dictionary containing the meta information about the sequence.
    """
    seqinfo_path = sequence_dir / 'seqinfo.ini'
    if not seqinfo_path.exists():
        raise FileNotFoundError(f'Sequence info ini in {sequence_dir} is missing.')

    seqinfo = configparser.ConfigParser()
    seqinfo.read(seqinfo_path)

    return {
        'name': seqinfo['Sequence']['name'],
        'len': int(seqinfo['Sequence']['seqLength']),
        'rangeBins': int(seqinfo['RAD']['rangeBins']),
        'azimuthBins': int(seqinfo['RAD']['azimuthBins']),
        'dopplerBins': int(seqinfo['RAD']['dopplerBins']),
        'imgWidth': int(seqinfo['Stereo']['imgWidth']),
        'imgHeight': int(seqinfo['Stereo']['imgHeight']),
    }


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


def convert_sequence(seq_name: str, split: str, seq_dir: Path, args: Namespace) -> Path:
    seqinfo = parse_sequence_info(seq_dir)
    radar_type = args.radar_type.lower()
    dataset_name = args.dataset_name.lower()
    new_seq_name = seq_name.replace(DATASET_NAME, dataset_name)
    out_dir: Path = args.out_dir / f'{dataset_name}-{split}' / new_seq_name
    seqinfo_path = out_dir / 'seqinfo.ini'
    img_dir = out_dir / 'img1'
    gt_dir = out_dir / 'gt'
    det_dir = out_dir / 'det'
    gt_file = gt_dir / 'gt.txt'
    det_file = det_dir / 'det.txt'

    # Convert radar data
    rd_w, rd_h = 0, 0
    min, max = 255, 0
    # 0.05-th quantile used to visualize min
    min_q = 255

    if not args.skip_radar:
        img_dir.mkdir(parents=True, exist_ok=True)
        for t in range(1, seqinfo['len'] + 1):
            rad = load_rad(seq_dir, t)

            if radar_type == 'rd':
                data = rad_to_rd(rad)
            elif radar_type == 'ra':
                data = rad_to_ra(rad)
            else:
                raise ValueError(f'Radar type \"{radar_type}\" not supported.')

            img = norm_to_image(data)
            assert len(img.shape) == 2
            rd_h, rd_w = data.shape

            # Min, max pixel values
            rd_min = np.min(img)
            rd_max = np.max(img)
            rd_q = np.quantile(img, 0.05)
            if rd_min < min:
                min = rd_min
            if rd_max > max:
                max = rd_max
            if rd_q < min_q:
                min_q = rd_q

            cv2.imwrite(str(img_dir / f'{t:06d}.png'), img)

    # Convert GT to MOT20 GT and Det
    gts = load_gt(seq_dir)

    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(det_dir, exist_ok=True)

    with open(gt_file, 'w', newline='') as gt_csv, open(det_file, 'w', newline='') as det_csv:
        gt_writer = csv.writer(gt_csv)
        det_writer = csv.writer(det_csv)

        for t, gt in enumerate(gts, start=1):
            for box, id in zip(gt['boxes'], gt['ids']):
                x, y, w, h = get_bb(box, radar_type)

                # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <consider_entry>, <class>, <visibility>
                gt_writer.writerow([t, id, x, y, w, h, 1, 1, 1])
                # <frame>, <<ignored>>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <<ignored>>, <<ignored>>
                det_writer.writerow([t, -1, x, y, w, h, 1, -1, -1])

    # Copy sequence info
    new_seqinfo = configparser.ConfigParser()
    new_seqinfo.optionxform = str # Preserve case
    new_seqinfo.add_section('Sequence')
    new_seqinfo.set('Sequence', 'name', new_seq_name)
    new_seqinfo.set('Sequence', 'imDir', img_dir.name)
    new_seqinfo.set('Sequence', 'imExt', '.png')
    new_seqinfo.set('Sequence', 'frameRate', '1')
    new_seqinfo.set('Sequence', 'seqLength', str(seqinfo['len']))
    if w > 0 and h > 0:
        new_seqinfo.add_section('Image')
        new_seqinfo.set('Image', 'width', str(rd_w))
        new_seqinfo.set('Image', 'height', str(rd_h))
        new_seqinfo.set('Image', 'min', str(min))
        new_seqinfo.set('Image', 'max', str(max))
        new_seqinfo.set('Image', 'minQ', str(int(min_q)))

    with open(seqinfo_path, 'w') as f:
        new_seqinfo.write(f, space_around_delimiters=False)

    return out_dir


def main(args: Namespace):
    sequence_dirs, splits = get_sequence_paths(args.data_dir)

    print('Copying sequence maps...')
    out_seqmaps_dir = args.out_dir / 'seqmaps'
    out_seqmaps_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset_name.lower()
    for split in splits:
        seqmap_file = args.data_dir / 'seqmaps' / f'{DATASET_NAME}-{split}.txt'
        new_seqmap_file = out_seqmaps_dir / f'{dataset_name}-{split}.txt'

        with open(seqmap_file, 'r') as fin, open(new_seqmap_file, 'w') as fout:
            for line in fin:
                fout.write(line.replace(DATASET_NAME, dataset_name))

    with tqdm_joblib(tqdm.tqdm(desc='Converting sequences', total=len(sequence_dirs))) as pbar:
        with joblib.Parallel(n_jobs=args.workers) as parallel:
            new_sequence_dirs = parallel(joblib.delayed(convert_sequence)(seq_name, split, seq_dir, args) for seq_name, split, seq_dir in sequence_dirs)

    print('Creating ALL split...')
    if args.create_all_split:
        all_split: Path = args.out_dir / f'{dataset_name}-all'

        if all_split.exists():
            shutil.rmtree(all_split)

        all_split.mkdir(parents=True)

        with open(out_seqmaps_dir / f'{dataset_name}-all.txt', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name'])

            for (seq_name, _, _), new_seq_dir in zip(sequence_dirs, new_sequence_dirs):
                new_seq_name = seq_name.replace(DATASET_NAME, dataset_name)
                seq_dir_link = all_split / new_seq_name
                seq_dir_link = seq_dir_link.resolve()

                if sys.platform == 'win32':
                    _winapi.CreateJunction(str(new_seq_dir.resolve()), str(seq_dir_link))
                else:
                    seq_dir_link.symlink_to(new_seq_dir.resolve(), target_is_directory=True)

                writer.writerow([new_seq_name])



if __name__ == '__main__':
    parser = ArgumentParser(description='Script to convert the RADTrack dataset to the MOT20 format.')
    parser.add_argument('--version', '-v', action='version', version='1.0.0')
    parser.add_argument('--workers', dest='workers', type=int, default=-1, help='Number of workers for parallel execution. -1 for all CPUs.')
    parser.add_argument('--data', '-d', dest='data_dir', type=Path, required=True, help='Path to the RADTrack dataset directory.')
    parser.add_argument('--output', '-o', dest='out_dir', type=Path, required=True, help='Path to the directory where the converted dataset should be saved.')
    parser.add_argument('--name', '-n', dest='dataset_name', type=str, required=True, help='Name of the new dataset.')
    parser.add_argument('--type', '-t', dest='radar_type', type=str, choices=['RD', 'RA'], default='RD', help='The output radar format. RD for Range-Doppler images and GT or RA for Range-Azimuth.')
    parser.add_argument('--skip-radar', '-s', dest='skip_radar', action='store_true', help='Skip converting the radar data. Only convert detections, GT and meta information.')
    parser.add_argument('--create-all-split', '-a', dest='create_all_split', action='store_true', help='Creates an additional -all split directory containing all sequences of the other splits as symbolic links.')

    args = parser.parse_args()

    main(args)

# python convert_to_mot20.py -d ..\data\dataset\RADTrack -t RD -n RDTrack -o ..\data\dataset\RDTrack -a
