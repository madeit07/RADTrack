import configparser
import csv
import json
import os

import cv2
import numpy as np

from util import util

class RADTrack:
    DATASET_NAME = 'radtrack'
    ALL_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
    TRACK_FIELDS = ['classes', 'boxes', 'cart_boxes', 'ids']

    def __init__(self, data_path: str):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'Directory \"{data_path}\" does not exist.')

        self._data_path = data_path
        self._radar_config = self._parseRadarConfig(data_path)
        self._sequences = self._parse_sequences(data_path, self._radar_config)

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

    def _parse_sequences(self, data_path: str, radar_config):
        sequence_paths = self.get_sequence_paths(data_path)
        return {seq_name: RADTrackSequence(seq_name, split, sequence_dir, radar_config) for seq_name, split, sequence_dir in sequence_paths}

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
    def __init__(self, name: str, split: str, sequence_dir: str, radar_config: dict):
        self._name = name
        self._split = split
        self.sequence_dir = sequence_dir
        self.radar_config = radar_config

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
        gt = self.gt[t - 1]

        if self.prediction:
            pred = self.prediction[t - 1] if (t - 1) < len(self.prediction) else self.get_default_track()
        else:
            pred = None

        stereo_file = self.stereo_file(t)
        stereo = self.load_stereo(stereo_file)
        image = util.get_left_image(stereo)

        rad_file = self.rad_file(t)
        rad = self.load_rad(rad_file)

        rd = util.rad_to_rd(rad)
        rdimg = util.norm_to_image(rd)

        ra = util.rad_to_ra(rad)
        raimg = util.norm_to_image(ra)

        cart = util.ra_to_cart(ra, self.radar_config, gap_fill=4)
        cartimg = util.norm_to_image(cart)

        return gt, pred, image, rdimg, raimg, cartimg


    def load_gt(self, file_path: str) -> list[dict[str]]:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            return []

    def load_prediction(self, file_path: str):
        if not os.path.exists(file_path):
            self._pred = None
        else:
            with open(file_path, 'r') as f:
                self._pred = json.load(f)

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

    def get_default_track(self):
        return {field: [] for field in RADTrack.TRACK_FIELDS}

    @property
    def gt(self):
        if self._gt is not None:
            return self._gt

        # Lazy load gt
        self._gt = self.load_gt(self.gt_file())
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

    @property
    def fps(self) -> int:
        return 10

    def __str__(self) -> str:
        return f'SEQ: {self.name} [{self.split}] (N_FRAMES: {len(self)})'

    def __repr__(self) -> str:
        return self.__str__()

