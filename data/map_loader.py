from torch.utils.data import DataLoader

from .obstacles import TrajectoryDataset, seq_collate

def data_loader(args, path, shuffle=True):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim,
        device=args.device,
        pixel_distance=args.pixel_distance,
        resize=args.map_size)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)
    return dset, loader
