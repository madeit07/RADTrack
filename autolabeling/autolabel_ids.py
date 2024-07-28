from argparse import ArgumentParser, Namespace
import contextlib
import datetime
import joblib
import os
import json
import time

import tqdm
import numpy as np

from util import dataset
from util.sort import Sort


def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x[...,0], x[...,1], x[...,2], x[...,3]
    b = [(cx - 0.5 * w), (cy - 0.5 * h),
         (cx + 0.5 * w), (cy + 0.5 * h)]
    return np.stack(b, axis=-1)


def label_sequence(seq_name: str, split: str, seq_dir: str, out_dir: str, args: Namespace):
    sort = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold)

    seq_len = dataset.get_sequence_len(seq_dir)
    gt_dir = os.path.join(seq_dir, 'gt')
    out_gt_dir = os.path.join(out_dir, f'{dataset.DATASET_NAME}-{split}', seq_name, 'gt')
    all_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']

    os.makedirs(out_gt_dir, exist_ok=True)

    gt_file = os.path.join(gt_dir, 'gt.json')
    out_gt_file = os.path.join(out_gt_dir, 'gt.json')

    gt: list[dict[str]] = []
    with open(gt_file, 'r') as f:
        gt = json.load(f)

    for i in range(seq_len):
        cart_xywh = np.array(gt[i]['cart_boxes'])
        cart_xyxy = box_cxcywh_to_xyxy(cart_xywh)
        classes = gt[i]['classes']
        cls_idx = np.array([all_classes.index(cls) for cls in classes]).reshape(-1, 1)

        certainties = np.ones((len(cart_xyxy), 1))
        order = np.arange(len(cart_xyxy)).reshape(-1, 1)
        cart_boxes = np.concatenate((cart_xyxy, certainties, order, cls_idx), axis=1)

        bbox_with_ids = sort.update(cart_boxes)

        # Sort ids by original order
        ids = np.full((len(cart_xyxy)), -1, dtype=np.int32)
        ids[bbox_with_ids[:, 5].astype(np.int32)] = bbox_with_ids[:, 4].astype(np.int32)

        gt[i]['ids'] = ids.tolist()

    with open(out_gt_file, 'w') as f:
        json.dump(gt, f, separators=(',', ':'))

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


def main(args: Namespace):
    start_time = time.time()

    print('Retrieving dataset information...')
    sequences = dataset.get_sequence_paths(args.data_dir)
    print(f'Found {len(sequences)} sequences.')

    with tqdm_joblib(tqdm.tqdm(desc='Labeling sequences', total=len(sequences))) as pbar:
        with joblib.Parallel(n_jobs=args.workers) as parallel:
            parallel(joblib.delayed(label_sequence)(*sequence, args.output_dir, args) for sequence in sequences)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Done. Total execution time: {total_time_str}')


if __name__ == '__main__':
    parser = ArgumentParser(description='Autolabel ids using Kalman Filter tracking on Cartesian ground truth data')
    parser.add_argument('--version', '-v', action='version', version='1.0.0')
    parser.add_argument('--data', '-d', dest='data_dir', type=str, required=True, help='Path to the RADTrack dataset directory.')
    parser.add_argument('--output', '-o', dest='output_dir', type=str, required=True, help='Path to the directory where the new GT data should be saved to.')
    parser.add_argument('--workers', dest='workers', type=int, default=-1, help='Number of workers for parallel execution. -1 for all CPUs.')
    parser.add_argument('--max-age', dest='max_age', type=int, default=5)
    parser.add_argument('--min-hits', dest='min_hits', type=int, default=0)
    parser.add_argument('--iou', dest='iou_threshold', type=float, default=0.1)

    args = parser.parse_args()

    main(args)

# python autolabel_ids.py -d ../data/dataset/RADTrack/ -o ../data/autolabeling/RADTrack
