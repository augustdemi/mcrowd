from torch.utils.data import DataLoader

from .trajectories import TrajectoryDataset, seq_collate
from .eth_trajectories import TrajectoryDataset as eth_Traj
from .eth_trajectories import seq_collate as eth_seq_collate

def data_loader(args, path, data_name='eth', data_split='train', shuffle=True):

    if 'Trajectories' in path:
        dset = TrajectoryDataset(
            path,
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            skip=args.skip,
            delim=args.delim,
            device=args.device)
        seq_col = seq_collate
    else:
        dset = eth_Traj(
            path,
            data_name=data_name,
            data_split=data_split,
            device=args.device)
        seq_col = eth_seq_collate


    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.loader_num_workers,
        collate_fn=seq_col)
    return dset, loader
