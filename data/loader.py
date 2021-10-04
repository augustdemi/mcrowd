from torch.utils.data import DataLoader

from .trajectories import TrajectoryDataset, seq_collate
from .sdd_trajectories import TrajectoryDataset as sdd_Traj
from .sdd_trajectories import seq_collate as sdd_seq_collate

def data_loader(args, path, data_split='train', shuffle=True):

    if 'Trajectories' in path:
        dset = TrajectoryDataset(
            path,
            data_split=data_split,
            device=args.device)
        seq_col = seq_collate
    else:
        dset = sdd_Traj(
            path,
            data_split=data_split,
            device=args.device)
        seq_col = sdd_seq_collate


    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.loader_num_workers,
        collate_fn=seq_col)
    return dset, loader
