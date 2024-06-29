import configparser
import csv
import os

DATASET_NAME = 'radtrack'

def get_sequence_len(sequence_dir: str) -> int:
    seqinfo_path = os.path.join(sequence_dir, 'seqinfo.ini')
    if not os.path.exists(seqinfo_path):
        raise FileNotFoundError(f'Sequence info ini in {sequence_dir} is missing.')

    seqinfo = configparser.ConfigParser()
    seqinfo.read(seqinfo_path)

    return int(seqinfo['Sequence']['seqLength'])


def get_sequence_paths(data_path: str) -> list[tuple[str, str, str]]:
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
