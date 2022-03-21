import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *, data_dir, deterministic=True
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_input_constraints = _list_input_files_recursively(data_dir)
    dataset = InputConstraintsDataset(
        all_input_constraints,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_input_files_recursively(data_dir):
    input_constraints = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npy"]:
            input_constraints.append(full_path)
        elif bf.isdir(full_path):
            input_constraints.extend(_list_input_files_recursively(full_path))
    return input_constraints


class InputConstraintsDataset(Dataset):
    def __init__(self, input_constraints_paths, shard=0, num_shards=1):
        super().__init__()
        self.local_input_constraints = input_constraints_paths[shard:][::num_shards]

    def __len__(self):
        return len(self.local_input_constraints)

    def __getitem__(self, idx):
        input_constraints_path = self.local_input_constraints[idx]
      
        input_constraints = np.load(input_constraints_path)
        print(input_constraints_path)
        return np.transpose(input_constraints, [2, 0, 1]).astype(np.float32)
