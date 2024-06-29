import contextlib
import glob
import os
import re
import shutil
import subprocess
from argparse import ArgumentParser, ArgumentTypeError, Namespace

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import joblib
import tqdm
import numpy as np

from util.dataloader import RADTrack, RADTrackSequence
from util import util


# https://stackoverflow.com/a/58936697
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
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

def get_rectangles(x: float, y: float, w: float, h: float, xmax: float, **kwargs):
    rectangles: list[patches.Rectangle] = []

    if x < 0:
        rectangles.append(patches.Rectangle((0, y), w + x, h, **kwargs))
        rectangles.append(patches.Rectangle((xmax + x, y), -x, h, **kwargs))
    elif (x + w) >= xmax:
        rectangles.append(patches.Rectangle((x, y), xmax - x, h, **kwargs))
        rectangles.append(patches.Rectangle((0, y), (x + w) - xmax, h, **kwargs))
    else:
        rectangles.append(patches.Rectangle((x, y), w, h, **kwargs))

    return rectangles

def plot_rd(ax, img):
    ax.set_title('Range-Doppler')
    ax.set_xticks([0, 16, 32, 48, 63])
    ax.set_xticklabels([-13, -6.5, 0, 6.5, 13])
    ax.set_yticks([0, 64, 128, 192, 255])
    ax.set_yticklabels([50, 37.5, 25, 12.5, 0])
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Range (m)")

    ax.imshow(img)

def plot_ra(ax, img):
    ax.set_title('Range-Azimuth')
    ax.set_xticks([0, 64, 128, 192, 255])
    ax.set_xticklabels([-85.87, -42.93, 0, 42.93, 85.87])
    ax.set_yticks([0, 64, 128, 192, 255])
    ax.set_yticklabels([50, 37.5, 25, 12.5, 0])
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Range (m)")

    ax.imshow(img)

def plot_cart(ax, img):
    ax.set_title('Cartesian')
    ax.set_xticks([0, 128, 256, 384, 512])
    ax.set_xticklabels([-50, -25, 0, 25, 50])
    ax.set_yticks([0, 64, 128, 192, 255])
    ax.set_yticklabels([50, 37.5, 25, 12.5, 0])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")

    ax.imshow(img)

def draw_bb(ax, box: np.ndarray, id: int, cls: str, shape: tuple[int], color, symmetrical: bool = False, is_pred: bool = False):
    yc, xc, h, w = box
    x, y = xc - w / 2, yc - h / 2
    ymax, xmax = shape

    if w * h <= 0:
        return

    rect_style = {
        'linewidth': 3,
        'edgecolor': color,
        'facecolor': 'none',
    }

    if is_pred:
        rect_style['linestyle'] = '--'
        rect_style['facecolor'] = color
        rect_style['alpha'] = 0.5

    if symmetrical:
        rects = get_rectangles(x, y, w, h, xmax, **rect_style)
    else:
        rects = [patches.Rectangle((x, y), w, h, **rect_style)]

    for r in rects:
        ax.add_patch(r)

    annotation_style = {
        'size': 12,
        'verticalalignment': 'baseline',
        'color': 'w',
        'backgroundcolor': 'none',
        # 'weight': 'bold',
        'bbox': {
            'facecolor': color,
            'alpha': 0.5,
            'pad': 2,
            'edgecolor': 'none',
        }
    }

    if is_pred:
        annotation_style['verticalalignment'] = 'top'
        annotation_style['horizontalalignment'] = 'right'
        annotation_style['bbox']['alpha'] = 0.75

        xtext = x + w
        if (x + w) > xmax:
            xtext = (x + w) - xmax
        ytext = y + 1
    else:
        xtext = 0 if x <= 0 or x + 25 >= xmax else x
        xtext += 1
        ytext = y - 2

    text = f'{id} {cls}' if cls else str(id)
    ax.annotate(text, (xtext, ytext), **annotation_style)


def plot(dataset: RADTrackSequence, frame: int, color_cycle: list, output_dir: str, args: Namespace):
    gt, pred, image, rd, ra, cart = dataset[frame]

    if args.only_stereo_rd:
        fig, axes = plt.subplots(ncols=2, width_ratios=[4, 2], figsize=(10, 8))
    else:
        fig, axes = plt.subplots(ncols=4, width_ratios=[6, 1, 5, 6], figsize=(36, 8))
    fig.tight_layout()

    axes[0].axis('off')
    axes[0].imshow(image)

    plot_rd(axes[1], rd)

    if not args.only_stereo_rd:
        plot_ra(axes[2], ra)
        plot_cart(axes[3], cart)

    if args.plot_gt:
        for i in range(len(gt['classes'])):
            cls = gt['classes'][i]
            bbox3d = np.array(gt['boxes'][i])
            cartBox = np.array(gt['cart_boxes'][i])
            ids = gt.get('ids', [])
            id = ids[i] if i < len(ids) else -1

            color = color_cycle[id % len(color_cycle)]

            rdBox = np.array([bbox3d[0], bbox3d[2], bbox3d[3], bbox3d[5]])
            raBox = np.array([bbox3d[0], bbox3d[1], bbox3d[3], bbox3d[4]])

            draw_bb(axes[1], rdBox, id, cls, rd.shape, color, symmetrical=True)

            if not args.only_stereo_rd:
                draw_bb(axes[2], raBox, id, cls, ra.shape, color, symmetrical=True)
                draw_bb(axes[3], cartBox, id, cls, cart.shape, color, symmetrical=False)

    if pred:
        for i in range(len(pred['classes'])):
            cls = pred['classes'][i]
            bbox3d = np.array(pred['boxes'][i])
            cartBox = np.array(pred['cart_boxes'][i])
            ids = pred.get('ids', [])
            id = ids[i] if i < len(ids) else -1

            color = color_cycle[id % len(color_cycle)]

            rdBox = np.array([bbox3d[0], bbox3d[2], bbox3d[3], bbox3d[5]])
            raBox = np.array([bbox3d[0], bbox3d[1], bbox3d[3], bbox3d[4]])

            draw_bb(axes[1], rdBox, id, cls, rd.shape, color, symmetrical=True, is_pred=True)

            if not args.only_stereo_rd:
                draw_bb(axes[2], raBox, id, cls, ra.shape, color, symmetrical=True, is_pred=True)
                draw_bb(axes[3], cartBox, id, cls, cart.shape, color, symmetrical=False, is_pred=True)

    out_file = os.path.join(output_dir, f'{frame:06d}.png')
    fig.savefig(out_file, dpi=args.dpi, bbox_inches="tight")

    plt.close(fig)


def visualize(dataset: RADTrackSequence, output_dir: str, color_cycle: list[str], args: Namespace):
    # If the frame index does not exist in the sequence the progress bar will show incomplete state
    if len(args.frames) == 0:
        total = len(dataset)
    else:
        total = len(args.frames)

    with tqdm_joblib(tqdm.tqdm(desc='Plotting frames', unit=' Frames', total=total, leave=False)) as pbar:
        with joblib.Parallel(n_jobs=args.workers) as parallel:
            parallel(joblib.delayed(plot)(dataset, frame, color_cycle, output_dir, args) for frame in range(1, len(dataset) + 1)
                     if len(args.frames) == 0 or frame in args.frames) # When user passes specific frames to plot, only plot these

    if args.create_video:
        fps = args.fps or dataset.fps

        scale = 900 if args.only_stereo_rd else 2500 # Image widths need to be divisible by 2

        subprocess.run(['ffmpeg',
                        '-r', str(fps),
                        '-i', os.path.join(output_dir, '%06d.png'),
                        '-c:v', 'libx264',
                        '-crf', '23',
                        '-preset', 'veryslow',
                        '-pix_fmt', 'yuv420p', # Needed to be supported by Windows
                        '-filter:v', f'scale={scale}:-1',
                        '-y', os.path.join(output_dir, f'{dataset.name}.mp4')])

        if args.remove_frames:
            for frame in glob.glob(os.path.join(output_dir, '*.png')):
                os.unlink(frame)


def clear_directory(folder: str):
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=False)
        except Exception as e:
            print(f'Failed to delete {path}. Reason: {e}')


def parse_frames_nums(argument: str):
    match = re.match(r'(\d+)(?:-(\d+))?$', argument)
    if not match:
        raise ArgumentTypeError(f'{argument} is not a range of number. Expected forms like \"0-5\" or \"2\".')

    start = int(match.group(1), 10)
    if start < 0:
        raise ArgumentTypeError(f'First number must be 0 or positive.')

    end = int(match.group(2) or str(start), 10)
    return set(range(start, end + 1))


def output_base_dir(split: str, args: Namespace) -> str:
    return os.path.join(args.output_dir, f'{RADTrack.DATASET_NAME}-{split}', args.output_name)


def main(args: Namespace):
    color_cycle = util.random_colors(n=20)

    # Convert sequence numbers to full sequence name
    args.sequences = [f'{RADTrack.DATASET_NAME}{num:04d}' for num in args.sequences]

    sequences: list[tuple[str, str, str]] = []
    seq_paths = RADTrack.get_sequence_paths(args.data_path)

    if args.sequences:
        sequences.extend(filter(lambda s: s[0] in args.sequences, seq_paths))
    else:
        sequences.extend(seq_paths)

    if not sequences:
        print('Nothing to visualize. Maybe forgot to specify correct dataset split?')
        return

    dataset = RADTrack(args.data_path)

    pbar = tqdm.tqdm(sequences, desc='Visualizing', unit='Sequence')
    for seq_name, split, seq_dir in pbar:
        seq_dataloader = dataset[seq_name]

        if args.tracks_path:
            track_name = seq_dataloader.name.replace(RADTrack.DATASET_NAME, args.tracks_prefix)
            prediction_file = os.path.join(args.tracks_path, f'{track_name}.json')
            if os.path.exists(prediction_file):
                seq_dataloader.load_prediction(prediction_file)

        pbar.set_description(f'Visualizing {seq_name}')
        pbar.set_postfix({'frames': len(dataset)})

        output_dir = os.path.join(output_base_dir(split, args), seq_name)
        os.makedirs(output_dir, exist_ok=True)

        if args.clear:
            clear_directory(output_dir)

        visualize(seq_dataloader, output_dir, color_cycle, args=args)

if __name__ == '__main__':
    parser = ArgumentParser(description=('Visualizes Range-Doppler maps by plotting heatmaps and converting the individual frames to a video. '
                                         'Additionally to the raw Range-Doppler data, CFAR detections, '
                                         'ground truth bounding boxes and bounding box predictions can be inserted.'))

    # Input
    inputgrp = parser.add_argument_group('Input')
    inputgrp.add_argument('--data', '-d', dest='data_path', type=str, required=True, help='Path to the directory containing the splits of the radar sequences.')
    inputgrp.add_argument('--sequences', '-s', dest='sequences', nargs='+', type=int, default=[], help='Specific sequences in the split to visualize. (Default is that all sequences in the data directory of given splits will be visualized.)')
    inputgrp.add_argument('--frames', '-f', dest='frame_sets', nargs='+', type=parse_frames_nums, default=[], metavar='RANGE', help='Only visualize given frame ranges (e.g. 0-100) of given sequences. (Default is that all frames will be visualized.)')

    # Track data
    trackgrp = parser.add_argument_group('Track', description='Track data settings to include in Range-Doppler maps.')
    trackgrp.add_argument('--tracks', '-t', dest='tracks_path', type=str, help='Path to the directory containing the tracker evaluation results. This will plot the track predictions.')
    trackgrp.add_argument('--tracks-prefix', '-tp', dest='tracks_prefix', type=str, default=RADTrack.DATASET_NAME, help='Prefix of the track files.')
    trackgrp.add_argument('--no-gt', '-ng', dest='plot_gt', action='store_false', help='Do not show the ground truth bounding boxes and id in the plot.')

    # Export options
    exportgrp = parser.add_argument_group('Export', description='Export settings for frames and video.')
    exportgrp.add_argument('--output', '-o', dest='output_dir', type=str, required=True, help='Path to the directory where the results should be saved.')
    exportgrp.add_argument('--name', '-n', dest='output_name', type=str, default='', help='Name of the directory where the results should be saved.')
    exportgrp.add_argument('--clear', '-c', dest='clear', action='store_true', help='Clears the output directory before writing to it.')
    exportgrp.add_argument('--video', '-vid', dest='create_video', action='store_true', help='Create a video from created plots. Does nothing if --frames is set.')
    exportgrp.add_argument('--remove-frames', '-rmf', dest='remove_frames', action='store_true', help='Deletes the frames after converting to video. Does nothing if --video flag was not set.')
    exportgrp.add_argument('--fps', '-r', dest='fps', type=int, default=10, help='Framerate of the resulting video. (Default is framerate specified by sequence.)')
    exportgrp.add_argument('--dpi', dest='dpi', type=int, default=72, help='Image DPI.')
    exportgrp.add_argument('--only-stereo-rd', dest='only_stereo_rd', action='store_true', help='Only plot RD and stereo.')

    # Misc
    miscgrp = parser.add_argument_group('Misc', description='General settings.')
    miscgrp.add_argument('--workers', dest='workers', type=int, default=4, help='Number of workers for parallel data fetching and processing. (Default: 4)')
    miscgrp.add_argument('--version', '-v', action='version', version='1.0.0')

    args = parser.parse_args()

    args.frames = set()
    args.frames.update(*args.frame_sets)

    if args.frames:
        if args.create_video:
            print('Creating video with frames out of order is unsupported.')
        args.create_video = False

    if not args.create_video:
        if args.remove_frames:
            print('Removing frames without creating a video is unsupported.')
        args.remove_frames = False

    main(args)

# Example 1: Visualize every sequence in the data folder belonging to train or val dataset
# python visualize.py --clear -d ../data/dataset/RADTrack -o ../data/vid --splits val train --video -rmf

# Example 2: Visualize sequence with number 76 and 23
# python visualize.py --clear -d ../data/dataset/RADTrack -o ../data/vid -s 76 23 --video -rmf

# python visualize.py --clear -d ../data/dataset/RADTrack -t ../data/trackers/rdtrack-val/run30_e79_ms5 -tp rdtrack -n run30_e79_ms5 -o ../data/vid -s 12 -f 101 --video --only-stereo-rd
