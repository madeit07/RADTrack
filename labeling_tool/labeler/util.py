import colorsys
import random
import numpy as np


def polar_to_cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return x, y


def rad_to_rd(rad: np.ndarray) -> np.ndarray:
    if rad is None:
        return None

    magnitude = pow(np.abs(rad), 2)
    rd = np.sum(magnitude, axis=1)
    rd = 10 * np.log10(rd + 1.)

    return rd


def rad_to_ra(rad: np.ndarray) -> np.ndarray:
    if rad is None:
        return None

    magnitude = pow(np.abs(rad), 2)
    ra = np.sum(magnitude, axis=-1)
    ra = 10 * np.log10(ra + 1.)

    return ra


def ra_to_cart(ra: np.ndarray, radar_config: dict[str], gap_fill: int = 1):
    if ra is None:
        return None

    rbins, abins = ra.shape
    output_mask = np.ones([rbins, rbins * 2]) * np.amin(ra)

    max_range = 50
    rres = radar_config['range_resolution']
    cf = radar_config['config_frequency']
    df = radar_config['designed_frequency']

    ranges = np.arange(rbins) * rres
    angles = np.arcsin((np.arange(abins) * 2 * np.pi / abins - np.pi) / (np.pi * cf / df))

    interp_ranges = np.interp(np.linspace(0, rbins, num=(rbins * gap_fill), endpoint=False), np.arange(rbins), ranges)
    interp_angles = np.interp(np.linspace(0, abins, num=(abins * gap_fill), endpoint=False), np.arange(abins), angles)

    rv, av = np.meshgrid(interp_ranges, interp_angles)
    zv, xv = polar_to_cart(rv, av)
    new_i = (rbins - np.round(zv / rres) - 1).astype(np.int64)
    new_j = (np.round((xv + max_range) / rres) - 1).astype(np.int64)

    i = np.repeat(np.arange(rbins), gap_fill)
    j = np.repeat(np.arange(abins), gap_fill)
    iv, jv = np.meshgrid(i, j)

    output_mask[new_i, new_j] = np.flip(ra, axis=0)[iv, jv]

    return output_mask


def get_left_image(stereo) -> np.ndarray:
    if stereo is None:
        return None
    return stereo[:, :stereo.shape[1]//2, ...][..., ::-1]


def norm_to_image(array: np.ndarray) -> np.ndarray:
    """Normalize to image format"""
    if array is None:
        return None

    # Normalized [0,1]
    n = (array - np.min(array)) / np.ptp(array)

    # Normalized [0,255]
    return (255 * n).astype(np.uint8)


def random_colors(n: int, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / n, 1., brightness) for i in range(n)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))

    random.seed(8888)
    random.shuffle(colors)

    return colors

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def numlist2str(numbers: list[int]) -> str:
    """Converts a list of numbers to range strings.

    Args:
        numbers (list[int]): List of numbers.

    Returns:
        str: Range strings, e.g. 0-5, 7, 9-12
    """
    ranges: list[str] = []
    numbers = sorted(numbers)
    start = numbers[0]
    prev = start
    for i in range(1, len(numbers) + 1):
        prev = numbers[i - 1]

        if i < len(numbers):
            cur = numbers[i]
        else:
            cur = None

        if cur != prev + 1:
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f'{start}-{prev}')
            start = cur

    return ', '.join(ranges)
