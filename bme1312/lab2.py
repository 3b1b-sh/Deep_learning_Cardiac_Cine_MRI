import itertools

from .utils import imgshow, imsshow, image_mask_overlay
from .utils import compute_num_params as compute_params
from .dataset import ImageFolder, Test_ImageFolder, get_loader

from .solver import Lab2Solver as Solver


def fetch_batch_sample(loader, idx):
    batch = next(itertools.islice(loader, idx, None))
    return batch
