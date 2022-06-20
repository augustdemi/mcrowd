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
     local_map, local_ic, local_homo, scale, congest_map) = zip(*data)
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
    map_path = np.concatenate(map_path, 0)
    inv_h_t = np.concatenate(inv_h_t, 0)
    # local_map = np.array(np.concatenate(local_map, 0))
    local_map = np.concatenate(local_map, 0)
    congest_map = np.concatenate(congest_map, 0)
    local_ic = np.concatenate(local_ic, 0)
    local_homo = torch.cat(local_homo, 0)


    obs_traj_st = obs_traj.clone()
    # pos is stdized by mean = last obs step
    obs_traj_st[:, :, :2] = (obs_traj_st[:,:,:2] - obs_traj_st[-1, :, :2]) / scale
    obs_traj_st[:, :, 2:] /= scale
    # print(obs_traj_st.max(), obs_traj_st.min())


    out = [
        obs_traj, fut_traj, obs_traj_st, fut_traj[:,:,2:4] / scale, seq_start_end,
        obs_frames, fut_frames, map_path, inv_h_t,
        local_map, local_ic, local_homo, congest_map
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



def get_congestion_local_map(map, all_traj, zoom=10, radius=19.2):
    radius = int(radius * zoom)
    context_size = radius * 2

    global_map = np.kron(map, np.ones((zoom, zoom)))
    expanded_obs_img = np.full((global_map.shape[0] + context_size, global_map.shape[1] + context_size),
        False, dtype=np.float32)
    expanded_obs_img[radius:-radius, radius:-radius] = global_map.astype(np.float32)  # 99~-99


    all_pixel = all_traj[:, [1, 0]] * zoom
    all_pixel = context_size // 2 + np.round(all_pixel).astype(int)

    local_map = expanded_obs_img[all_pixel[7,0] - radius: all_pixel[7,0] + radius,
                all_pixel[7,1] - radius: all_pixel[7,1] + radius]
    return 1-local_map/255


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

        self.num_seq = len(self.seq_start_end) # = slide (seq. of 16 frames) 수 = 2692
        print(self.seq_start_end[-1])



    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        # global_map = imageio.imread(self.map_file_name[index])
        # map_file_name = self.map_file_name[index].replace('../../datasets', 'C:/dataset')
        map_file_name = self.map_file_name.replace('Trajectories', 'Y')
        global_map = imageio.imread(map_file_name)

        congest_maps = []
        for idx in range(start, end):
            all_traj = np.concatenate([self.obs_traj[idx, :2], self.fut_traj[idx, :2]], axis=1).transpose(1, 0)
            '''
            plt.imshow(global_map)
            plt.scatter(all_traj[:8,0], all_traj[:8,1], s=1, c='b')
            plt.scatter(all_traj[8:,0], all_traj[8:,1], s=1, c='r')
            plt.show()
            '''
            congest_maps.append(get_congestion_local_map(global_map, all_traj, zoom=10, radius=9.6))
            '''
            env = 1 - self.local_map[idx][0]
            c  = get_congestion_local_map(global_map, all_traj, zoom=10, radius=9.6)
            plt.imshow(np.stack([env, env * c, env], axis=2))
            '''
        congest_maps = np.stack(congest_maps)



        #########
        out = [
            self.obs_traj[start:end, :], self.fut_traj[start:end, :],
            self.obs_frame_num[start:end], self.fut_frame_num[start:end],
            np.array([self.map_file_name[index]] * (end - start)), np.array([self.inv_h_t[index]] * (end - start)),
            self.local_map[start:end],
            self.local_ic[start:end],
            torch.from_numpy(self.local_homo[start:end]).float().to(self.device), self.scale, congest_maps
        ]
        return out
