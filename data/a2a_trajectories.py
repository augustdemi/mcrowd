import logging
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle5
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import imageio
from utils import derivative_of

logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list,
     obs_frames, fut_frames, map_path, inv_h_t,
     local_map, local_ic, local_homo, scale) = zip(*data)
    scale = scale[0]

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    fut_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    seq_start_end = torch.LongTensor(seq_start_end)

    obs_frames = np.concatenate(obs_frames, 0)
    fut_frames = np.concatenate(fut_frames, 0)
    map_path = np.stack(map_path, 0)
    inv_h_t = np.stack(inv_h_t, 0)
    # local_map = np.array(np.concatenate(local_map, 0))
    local_ic = np.concatenate(local_ic, 0)
    local_homo = torch.cat(local_homo, 0)

    local_maps = []
    for i in range(len(seq_start_end)):
        local_maps.extend(local_map[i])

    obs_traj_st = obs_traj.clone()
    # pos is stdized by mean = last obs step
    obs_traj_st[:, :, :2] = (obs_traj_st[:,:,:2] - obs_traj_st[-1, :, :2]) / scale
    obs_traj_st[:, :, 2:] /= scale
    # print(obs_traj_st.max(), obs_traj_st.min())


    out = [
        obs_traj, fut_traj, obs_traj_st, fut_traj[:,:,2:4] / scale, seq_start_end,
        obs_frames, fut_frames, map_path, inv_h_t,
        local_maps, local_ic, local_homo
    ]


    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)



class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, data_split, device='cpu', scale=1
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = os.path.join(data_dir, data_split)
        self.device = device
        self.scale = scale
        self.obs_len = 8
        self.dt = 0.4

        with open(os.path.join(data_dir, data_split + '.pkl'), 'rb') as handle:
            all_data = pickle5.load(handle)


        self.obs_frame_num = all_data['obs_frame_num']
        self.fut_frame_num = all_data['fut_frame_num']

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(all_data['obs_traj']).float().to(self.device)
        self.fut_traj = torch.from_numpy(all_data['fut_traj']).float().to(self.device)

        self.seq_start_end = np.array(all_data['seq_start_end'])
        self.data_file_name = all_data['data_file_name']
        self.inv_h_t = all_data['inv_h_t']

        self.local_map = all_data['local_map']
        self.local_homo = torch.from_numpy(all_data['local_homo']).float().to(self.device)
        self.local_ic = all_data['local_ic']

        self.num_seq = len(self.seq_start_end)
        print(self.seq_start_end[-1])


    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        '''
        start, end = index, index+1
        seq_idx = np.where((index >= self.seq_start_end[:,0]) & (index < self.seq_start_end[:,1]))[0][0]
        plt.imshow(self.local_map[index])
        all_pixel_local = self.local_ic[index]
        plt.scatter(all_pixel_local[:8, 1], all_pixel_local[:8, 0], s=1, c='b')
        plt.scatter(all_pixel_local[8:, 1], all_pixel_local[8:, 0], s=1, c='r')
        plt.show()
        '''

        '''
        all_traj = self.obs_traj[index,:2].to(self.device).transpose(1,0)
        map_traj = np.matmul(np.concatenate([all_traj, np.ones((len(all_traj), 1))], 1), self.inv_h_t['/'.join(self.data_file_name[seq_idx].split('/')[:-1])])
        map_traj /= np.expand_dims(map_traj[:, 2], 1)
        map_traj = map_traj[:, :2]

        global_map = cv2.imread(os.path.join('C:\dataset\HTP-benchmark\A2A Data', self.data_file_name[seq_idx].replace('txt', 'png')))
        plt.imshow(global_map)
        plt.scatter(map_traj[:8, 0], map_traj[:8, 1], s=1, c='b')
        plt.scatter(map_traj[8:, 0], map_traj[8:, 1], s=1, c='r')
        '''
        #########
        out = [
            self.obs_traj[start:end].to(self.device), self.fut_traj[start:end].to(self.device),
            self.obs_frame_num[start:end], self.fut_frame_num[start:end],
            self.data_file_name[index], self.inv_h_t['/'.join(self.data_file_name[index].split('/')[:-1])],
            self.local_map[start:end],
            self.local_ic[start:end],
            self.local_homo[start:end],
            self.scale
        ]
        return out
