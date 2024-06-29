from argparse import ArgumentParser, Namespace
import contextlib
import os
import shutil
import glob

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

def main(args: Namespace):
    src_files = glob.glob(os.path.join(args.src_dir, '**', 'gt.json'), recursive=True)
    common_suffix = [os.path.relpath(f, args.src_dir) for f in src_files]
    dst_files = [os.path.join(args.dst_dir, f) for f in common_suffix]

    with tqdm_joblib(tqdm.tqdm(desc='Replacing GT files', total=len(src_files))) as pbar:
        with joblib.Parallel(n_jobs=args.workers) as parallel:
            parallel(joblib.delayed(shutil.copy2)(src, dst) for src, dst in zip(src_files, dst_files))


if __name__ == '__main__':
    parser = ArgumentParser(description='Script to replace RADTrack GT files with auto labeling or backup GT files.')
    parser.add_argument('--version', '-v', action='version', version='1.0.0')
    parser.add_argument('--dst', '-d', dest='dst_dir', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--src', '-s', dest='src_dir', type=str, required=True, help='Path to the directory containing the source files.')
    parser.add_argument('--workers', dest='workers', type=int, default=-1, help='Number of workers for parallel execution. -1 for all CPUs.')

    args = parser.parse_args()

    main(args)

# python replace_gt.py --src ..\data\autolabeling\RADTrack --dst ..\data\dataset\RADTrack
# python replace_gt.py --src ..\data\autolabeling_stereo\RADTrack --dst ..\data\dataset\RADTrack
