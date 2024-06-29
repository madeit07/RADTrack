from argparse import ArgumentParser, Namespace
import contextlib
import csv
import json
import os
import pickle

import joblib
import tqdm
import numpy as np


DATASET_NAME = 'radtrack'

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


def get_sequence_paths(data_path: str) -> list[tuple[str, str, str]]:
    """Returns the paths to all sequences in the dataset.

    Args:
        data_path (str): Path to the dataset containing the splits and sequences.

    Returns:
        list[tuple[str, str, str]]: List of sequence names, its split and path
    """
    seqmaps_dir = os.path.join(data_path, 'seqmaps')
    seqmaps = [f.name for f in os.scandir(seqmaps_dir) if f.is_file()]
    splits = [os.path.splitext(f)[0].split('-')[-1] for f in seqmaps]

    # seq_name, split, sequence_dir
    sequences: list[tuple[str, str, str]] = []
    for split, seqmap in zip(splits, seqmaps):
        split_dir = os.path.join(data_path, f'{DATASET_NAME}-{split}')

        with open(os.path.join(seqmaps_dir, seqmap), 'r', newline='') as f:
            seq_reader = csv.reader(f)
            sequences.extend([(seq[0], split, os.path.join(split_dir, seq[0])) for seq in seq_reader if seq[0] != 'name'])

    return sequences

def get_pickle_files(dir: str):
    return [f.name for f in os.scandir(dir) if f.is_file() and f.name.endswith('.pickle')]

def merge_gt(seq_dir: str):
    gt_dir = os.path.join(seq_dir, 'gt')
    gt_filenames = sorted(get_pickle_files(gt_dir))

    if len(gt_filenames) == 0:
        return

    # Gather all gt objects
    gts: list[dict[str]] = []
    for gt_filename in gt_filenames:
        with open(os.path.join(gt_dir, gt_filename), 'rb') as f:
            gt: dict[str] = pickle.load(f)
            gt_json = {}
            for key, value in gt.items():
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                gt_json[key] = value

            gts.append(gt_json)


    assert len(gt_filenames) == len(gts)

    # Write GT json
    with open(os.path.join(gt_dir, 'gt.json'), 'w') as f:
        json.dump(gts, f)

    # Remove old gt files
    for gt_filename in gt_filenames:
        os.remove(os.path.join(gt_dir, gt_filename))


def main(args: Namespace):
    sequence_dirs = get_sequence_paths(args.data_dir)

    with tqdm_joblib(tqdm.tqdm(desc='Merging GT files', total=len(sequence_dirs))) as pbar:
        with joblib.Parallel(n_jobs=args.workers) as parallel:
            parallel(joblib.delayed(merge_gt)(seq_dir) for _, _, seq_dir in sequence_dirs)


if __name__ == '__main__':
    parser = ArgumentParser(description='Script to merge RADTrack GT files into one large json file.')
    parser.add_argument('--version', '-v', action='version', version='1.0.0')
    parser.add_argument('--data', '-d', dest='data_dir', type=str, required=True, help='Path to the RADDet dataset directory.')
    parser.add_argument('--workers', dest='workers', type=int, default=-1, help='Number of workers for parallel execution. -1 for all CPUs.')

    args = parser.parse_args()

    main(args)

# python merge_gt.py -d ..\data\dataset\RADTrack
