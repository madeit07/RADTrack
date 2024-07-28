import configparser
import csv
import json
import os
import logging

import cv2
import numpy as np

from labeler import util

logger = logging.getLogger(__name__)

class RADTrack:
    DATASET_NAME = 'radtrack'
    ALL_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
    TRACK_FIELDS = ['classes', 'boxes', 'cart_boxes', 'ids']

    def __init__(self, data_path: str, cache_path: str = '.cache'):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'Directory \"{data_path}\" does not exist.')

        self._cache_path = cache_path
        self._data_path = data_path
        self._radar_config = self._parseRadarConfig(data_path)
        self._sequences = self._parse_sequences(data_path, self._radar_config, cache_path)

    @property
    def sequences(self):
        return self._sequences

    @property
    def data_dir(self):
        return self._data_path

    @property
    def radar_config(self) -> dict[str]:
        return self._radar_config

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, name: str):
        return self._sequences[name]

    def _parseRadarConfig(self, data_path: str) -> dict[str]:
        file_path = os.path.join(data_path, 'sensors_para', 'radar_config.json')
        with open(file_path, 'r') as f:
            return json.load(f)

    def _parse_sequences(self, data_path: str, radar_config, cache_path: str):
        sequence_paths = self.get_sequence_paths(data_path)
        return {seq_name: RADTrackSequence(seq_name, split, sequence_dir, radar_config, os.path.join(cache_path, seq_name)) for seq_name, split, sequence_dir in sequence_paths}

    @staticmethod
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
            split_dir = os.path.join(data_path, f'{RADTrack.DATASET_NAME}-{split}')

            with open(os.path.join(seqmaps_dir, seqmap), 'r', newline='') as f:
                seq_reader = csv.reader(f)
                sequences.extend([(seq[0], split, os.path.join(split_dir, seq[0])) for seq in seq_reader if seq[0] != 'name'])

        return sequences

    @staticmethod
    def parse_sequence_info(sequence_dir: str) -> dict[str, int | str]:
        """Returns all meta information about the sequence.

        Args:
            sequence_dir (str): Path to the sequence directory.

        Raises:
            FileNotFoundError: Sequence info ini is missing.

        Returns:
            dict[str, int | str]: Dictionary containing the meta information about the sequence.
        """
        seqinfo_path = os.path.join(sequence_dir, 'seqinfo.ini')
        if not os.path.exists(seqinfo_path):
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


class RADTrackSequence:
    def __init__(self, name: str, split: str, sequence_dir: str, radar_config: dict, cache_dir: str):
        self._name = name
        self._split = split
        self.sequence_dir = sequence_dir
        self.radar_config = radar_config
        self._cache_dir = cache_dir

        self._gt: list[dict[str]] | None = None
        self._pred: list[dict[str]] | None = None
        self._info = RADTrack.parse_sequence_info(sequence_dir)

    def gt_file(self) -> str:
        return os.path.join(self.sequence_dir, 'gt', 'gt.json')

    def stereo_file(self, t: int) -> str:
        return os.path.join(self.sequence_dir, 'stereo_image', f'{t:06d}.jpg')

    def rad_file(self, t: int) -> str:
        return os.path.join(self.sequence_dir, 'RAD', f'{t:06d}.npy')

    def __len__(self) -> int:
        return self._info['len']

    def __getitem__(self, t: int):
        return self._get_frame(t)

    def _get_frame(self, t: int):
        # Frame t is 1-based
        gt = self.gt[t - 1] if (t - 1) < len(self.gt) else self.get_default_track()
        if self.prediction:
            pred = self.prediction[t - 1] if (t - 1) < len(self.prediction) else self.get_default_track()
        else:
            pred = None

        stereo_file = self.stereo_file(t)
        stereo = self.load_stereo(stereo_file)

        image = util.get_left_image(stereo)
        rdimg, raimg, cartimg = self._get_rad_from_cache(t)

        return gt, pred, image, rdimg, raimg, cartimg

    def _get_rad_from_cache(self, t: int):
        self.cache(t)

        rdimg = self._get_from_cache('RD', t)
        raimg = self._get_from_cache('RA', t)
        cartimg = self._get_from_cache('Cart', t)

        return rdimg, raimg, cartimg

    def _get_from_cache(self, radtype: str, t: int) -> np.ndarray | None:
        file = os.path.join(self._cache_dir, radtype, f'{t:06d}.npy')

        if not os.path.exists(file):
            return None

        # Retry 2 times
        for i in range(1, 3):
            try:
                return np.load(file)
            except Exception as e:
                # Delete file and rebuild cache
                logger.warning(f'Could not load \"{file}\" from cache ({i}. try). Rebuilding cache file. {e} ({self} radtype={radtype}, t={t})')
                try:
                    os.remove(file)
                except:
                    pass
                self.cache(t)

        logger.error(f'Could not load \"{file}\" from cache after rebuilding. ({self} radtype={radtype}, t={t})')

        return None

    def _save_to_cache(self, radtype: str, t: int, image: np.ndarray):
        if image is None:
            logger.warning(f'Invalid images cannot be saved to cache. ({self} radtype={radtype}, t={t})')
            return

        file = os.path.join(self._cache_dir, radtype, f'{t:06d}.npy')
        os.makedirs(os.path.dirname(file), exist_ok=True)
        np.save(file, image)

    def cache(self, t: int):
        filename = f'{t:06d}.npy'
        rd_file = os.path.join(self._cache_dir, 'RD', filename)
        ra_file = os.path.join(self._cache_dir, 'RA', filename)
        cart_file = os.path.join(self._cache_dir, 'Cart', filename)

        rd_exists = os.path.exists(rd_file)
        ra_exists = os.path.exists(ra_file)
        cart_exists = os.path.exists(cart_file)

        rad = None
        if (not rd_exists) or (not ra_exists) or (not cart_exists):
            rad_file = self.rad_file(t)
            rad = self.load_rad(rad_file)

        if not rd_exists:
            rd = util.rad_to_rd(rad)
            rdimg = util.norm_to_image(rd)
            self._save_to_cache('RD', t, rdimg)

        ra = None
        if not ra_exists:
            ra = util.rad_to_ra(rad)
            raimg = util.norm_to_image(ra)
            self._save_to_cache('RA', t, raimg)

        if not cart_exists:
            if ra is None:
                ra = util.rad_to_ra(rad)

            cart = util.ra_to_cart(ra, self.radar_config, gap_fill=4)
            cartimg = util.norm_to_image(cart)
            self._save_to_cache('Cart', t, cartimg)

    def load_gt(self, file_path: str) -> list[dict[str]]:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            return []

    def load_rad(self, file_path: str) -> np.ndarray | None:
        if os.path.exists(file_path):
            return np.load(file_path)
        else:
            return None

    def load_stereo(self, file_path: str) -> np.ndarray | None:
        if os.path.exists(file_path):
            return cv2.imread(file_path)
        else:
            return None

    def cache_gt(self, gt: list[dict[str]]):
        self._gt = gt

    def save_gt(self):
        gt_file = self.gt_file()
        with open(gt_file, 'w') as f:
            json.dump(self.gt, f, separators=(',', ':'))

    def load_prediction(self, file_path: str):
        if not os.path.exists(file_path):
            self._pred = None
        else:
            with open(file_path, 'r') as f:
                self._pred = json.load(f)

    def get_default_track(self):
        return {field: [] for field in RADTrack.TRACK_FIELDS}

    @property
    def gt(self):
        if self._gt is not None:
            return self._gt

        # Lazy load gt
        self.cache_gt(self.load_gt(self.gt_file()))
        return self._gt

    @property
    def prediction(self):
        return self._pred

    @property
    def name(self) -> str:
        return self._name

    @property
    def split(self) -> str:
        return self._split

    @property
    def full_split(self) -> str:
        return f'{RADTrack.DATASET_NAME}-{self.split}'

    def __str__(self) -> str:
        return f'SEQ: {self.name} [{self.split}] (N_FRAMES: {len(self)})'

    def __repr__(self) -> str:
        return self.__str__()
