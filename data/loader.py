from torch.utils.data import DataLoader

from .trajectories import TrajectoryDataset, seq_collate

def data_loader(args, path, shuffle=True, map_ae=False):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim,
        device=args.device,
        resize=args.map_size,
        map_ae=map_ae)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)
    return dset, loader
