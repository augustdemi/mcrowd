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

        self.seq_start_end = all_data['seq_start_end']
        self.map_file_name = all_data['map_file_name']
        self.inv_h_t = all_data['inv_h_t']

        self.local_map = np.expand_dims(all_data['local_map'],1)
        self.local_homo = all_data['local_homo']
        self.local_ic = all_data['local_ic']

        self.num_seq = len(self.local_ic) # = slide (seq. of 16 frames) 수 = 2692
        print(self.seq_start_end[-1])



    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):

        #########
        out = [
            self.obs_traj[index, :], self.fut_traj[index, :],
            self.local_map[index],
            self.local_ic[index],
            torch.from_numpy(self.local_homo[index]).float().to(self.device)
        ]
        return out
