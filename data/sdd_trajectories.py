import logging
import os
import math
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import derivative_of

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import imageio
from skimage.transform import resize
import pickle

logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list,
     obs_frames, fut_frames, map_path, inv_h_t,
     local_map, local_ic, local_homo) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    fut_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    seq_start_end = torch.LongTensor(seq_start_end)

    obs_frames = np.stack(obs_frames)
    fut_frames = np.stack(fut_frames)
    # map_path = np.concatenate(map_path, 0)
    inv_h_t = np.concatenate(inv_h_t, 0)
    # local_map = np.array(np.concatenate(local_map, 0))
    # local_map = np.concatenate(local_map, 0)
    local_ic = np.concatenate(local_ic, 0)
    local_homo = np.stack(local_homo, 0)


    out = [
        obs_traj, fut_traj, seq_start_end,
        obs_frames, fut_frames, map_path, inv_h_t,
        local_map, local_ic, local_homo
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


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def transform(image, resize):
    im = Image.fromarray(image[0])

    image = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])(im)
    return image


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_dir, data_split, device='cpu'
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

        # self.data_dir = '../../datasets/eth/test'
        self.obs_len = 8
        self.pred_len = 12
        self.skip = 1
        self.seq_len = self.obs_len + self.pred_len
        self.delim = ' '
        self.device = device
        self.map_dir = os.path.join(data_dir, 'SDD_semantic_maps', data_split + '_masks')
        self.data_path = os.path.join(data_dir, data_split + '_processed.pkl')

        self.seq_len = self.obs_len + self.pred_len

        self.maps={}
        for file in os.listdir(self.map_dir):
            self.maps.update({file.split('.')[0]:imageio.imread(os.path.join(self.map_dir, file))})

        with open(self.data_path, 'rb') as handle:
            all_data = pickle.load(handle)



        self.obs_frame_num = all_data['obs_frame_num']
        self.fut_frame_num = all_data['fut_frame_num']

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(all_data['obs_traj']).float().to(self.device)
        self.fut_traj = torch.from_numpy(all_data['fut_traj']).float().to(self.device)

        self.seq_start_end = all_data['seq_start_end']
        self.map_file_name = all_data['map_file_name']
        self.inv_h_t = all_data['inv_h_t']

        self.local_ic = all_data['local_ic']
        self.obs_heatmap = np.expand_dims(all_data['obs_heatmap'],1)
        self.fut_heatmap = np.expand_dims(all_data['fut_heatmap'],1)

        print(self.seq_start_end[-1])


    def __len__(self):
        return len(self.obs_traj)

    def __getitem__(self, index):
        global_map = np.expand_dims(self.maps[self.map_file_name[index] + '_mask'], axis=0)
        inv_h_t = np.expand_dims(np.eye(3), axis=0)
        local_ics = torch.cat([self.obs_traj[index, :2, :],  self.fut_traj[index, :2, :]], dim=1)[[1,0],:].detach().cpu().numpy()
        local_ics = np.round(local_ics).astype(int).transpose((1,0))

        #########
        out = [
            self.obs_traj[index].to(self.device).unsqueeze(0), self.fut_traj[index].to(self.device).unsqueeze(0),
            self.obs_frame_num[index], self.fut_frame_num[index],
            global_map, inv_h_t,
            global_map, np.expand_dims(local_ics, axis=0), inv_h_t
        ]
        return out
