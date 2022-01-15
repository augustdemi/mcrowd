from torch.utils.data import DataLoader

from .trajectories import TrajectoryDataset, seq_collate
from .sdd_trajectories import TrajectoryDataset as sdd_Traj
from .sdd_trajectories import seq_collate as sdd_seq_collate

from .kitti_trajectories import TrackDataset as kitti_Traj
from .kitti_trajectories import seq_collate as kitti_seq_collate


from .a2a_trajectories import TrajectoryDataset as a2a_Traj
from .a2a_trajectories import seq_collate as a2a_seq_collate

def data_loader(args, path, data_split='train', shuffle=True):

    if 'Trajectories' in path:
        dset = TrajectoryDataset(
            path,
            data_split=data_split,
            device=args.device)
        seq_col = seq_collate
    # elif 'Nuscenes' in path:
    #     generator
    elif 'KITTI' in path:
        dset = kitti_Traj(
            path,
            data_split=data_split,
            device=args.device)
        seq_col = kitti_seq_collate
    elif 'A2A' in path:
        dset = a2a_Traj(
            path,
            data_split=data_split,
            device=args.device)
        seq_col = a2a_seq_collate
    else:
        dset = sdd_Traj(
            path,
            data_split=data_split,
            device=args.device,
            scale=args.scale)
        seq_col = sdd_seq_collate


    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.loader_num_workers,
        collate_fn=seq_col)
    return dset, loader
