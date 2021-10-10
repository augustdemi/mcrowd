import logging
import os
import math
import pandas as pd

import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.append('../')
from utils import derivative_of
from scipy import ndimage
import cv2
import pickle

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import imageio


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

def get_local_map_ic(map, all_traj, zoom=10, radius=8):
    radius = radius * zoom
    context_size = radius * 2

    global_map = np.kron(map, np.ones((zoom, zoom)))
    expanded_obs_img = np.full((global_map.shape[0] + context_size, global_map.shape[1] + context_size),
        False, dtype=np.float32)
    expanded_obs_img[radius:-radius, radius:-radius] = global_map.astype(np.float32)  # 99~-99


    all_pixel = all_traj[:, [1, 0]] * zoom
    all_pixel = context_size // 2 + np.round(all_pixel).astype(int)

    # plt.imshow(expanded_obs_img)
    # plt.scatter(all_pixel[:8, 1], all_pixel[:8, 0], s=1, c='b')
    # plt.scatter(all_pixel[8:, 1], all_pixel[8:, 0], s=1, c='r')
    # plt.show()


    local_map = expanded_obs_img[all_pixel[7,0] - radius: all_pixel[7,0] + radius,
                all_pixel[7,1] - radius: all_pixel[7,1] + radius]

    '''
    for i in range(len(all_traj)):
        expanded_obs_img[all_pixel[i, 0], all_pixel[i, 1]] = i+10

    pts_ic = []
    for i in range(len(all_traj)):
        pts_ic.append([np.where(local_map == i+10)[0][0], np.where(local_map == i+10)[1][0]])
        expanded_obs_img[all_pixel[i, 0], all_pixel[i, 1]] = 255
    plt.imshow(local_map)
    plt.show()
    for p in pts_ic:
        plt.scatter(p[1], p[0], s=1, c='r')


    h, _ = cv2.findHomography(np.array(pts_ic), all_traj)
    '''

    fake_pt = [all_traj[7]]
    for i in range(1, 6):
        fake_pt.append(all_traj[7] + [i, i] + np.random.rand(2)*0.3)
        fake_pt.append(all_traj[7] + [-i, -i] + np.random.rand(2)*0.3)
        fake_pt.append(all_traj[7] + [i, -i] + np.random.rand(2)*0.3)
        fake_pt.append(all_traj[7] + [-i, i] + np.random.rand(2)*0.3)
    fake_pt = np.array(fake_pt)
    fake_pixel = fake_pt[:,[1, 0]] * zoom
    fake_pixel = radius + np.round(fake_pixel).astype(int)

    temp_map_val = []
    for i in range(len(fake_pixel)):
        temp_map_val.append(expanded_obs_img[fake_pixel[i, 0], fake_pixel[i, 1]])
        expanded_obs_img[fake_pixel[i, 0], fake_pixel[i, 1]] = i + 10


    fake_local_pixel = []
    for i in range(len(fake_pixel)):
        fake_local_pixel.append([np.where(local_map == i + 10)[0][0], np.where(local_map == i + 10)[1][0]])
        expanded_obs_img[fake_pixel[i, 0], fake_pixel[i, 1]] = temp_map_val[i]

    h, _ = cv2.findHomography(np.array([fake_local_pixel]), np.array(fake_pt))

    # plt.scatter(np.array(fake_local_pixel)[:, 1], np.array(fake_local_pixel)[:, 0], s=1, c='g')

    ## validate
    all_pixel_local = np.matmul(np.concatenate([all_traj, np.ones((len(all_traj), 1))], axis=1),
                                np.linalg.pinv(np.transpose(h)))
    all_pixel_local /= np.expand_dims(all_pixel_local[:, 2], 1)
    all_pixel_local = np.round(all_pixel_local).astype(int)[:,:2]

    # back to wc
    # back_wc = np.matmul(np.concatenate([all_pixel_local, np.ones((len(all_pixel_local), 1))], axis=1), np.transpose(h))
    # back_wc /= np.expand_dims(back_wc[:, 2], 1)
    # back_wc = back_wc[:,:2]

    #
    # plt.imshow(local_map)
    # plt.scatter(all_pixel_local[:8, 1], all_pixel_local[:8, 0], s=1, c='b')
    # plt.scatter(all_pixel_local[8:, 1], all_pixel_local[8:, 0], s=1, c='r')
    # plt.show()
    # per_step_pixel = np.sqrt(((all_pixel_local[1:] - all_pixel_local[:-1]) ** 2).sum(1)).mean()
    # per_step_wc = np.sqrt(((all_traj[1:] - all_traj[:-1]) ** 2).sum(1)).mean()


    # local_map = transforms.Compose([
    #     transforms.ToTensor()
    # ])(Image.fromarray(1-local_map/255))

    return 1-local_map/255, all_pixel_local, h


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
            self, data_dir, data_split, obs_len=8, pred_len=12, skip=1,
            min_ped=0, delim=',', dt=0.4
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

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        n_state=6
        # root_dir = '/dresden/users/ml1323/crowd/datasets/Trajectories'
        # root_dir = os.path.join(data_dir, data_name)
        # root_dir = 'D:\crowd\datasets\Trajectories\Trajectories'

        all_files = [e for e in os.listdir(data_dir) if ('.csv' in e) and ('homo' not in e)]
        all_files = np.array(sorted(all_files, key=lambda x: int(x.split('.')[0])))

        if data_split == 'train':
            all_files = all_files[:40]
            per_agent=5
            num_data=50
        elif data_split == 'val':
            all_files = all_files[[42,44]]
            per_agent=10
            num_data= 50
        else:
            all_files = all_files[[43,47,48,49]]
            per_agent=20
            num_data = 50

        # with open(os.path.join(root_dir, 'exit_wc.json')) as data_file:
        #     all_exit_wc = json.load(data_file)


        all_data = []
        for path in all_files:
            # exit_wc = np.array(all_exit_wc[path])
            scene_name = path.split('.')[0]
            path = os.path.join(data_dir, path.rstrip().replace('\\', '/'))
            print('data path:', path)
            # if 'Pathfinding' not in path:
            #     continue
            map_file_name = path.replace('.csv', '.png')
            print('map path: ', map_file_name)
            h = np.loadtxt(path.replace('.csv', '_homography.csv'), delimiter=',')
            inv_h_t = np.linalg.pinv(np.transpose(h))

            loaded_data = read_file(path, delim)

            data1 = pd.DataFrame(loaded_data)
            data1.columns = ['f', 'a', 'pos_x', 'pos_y']
            # data.sort_values(by=['f', 'a'], inplace=True)
            data1.sort_values(by=['f', 'a'], inplace=True)

            uniq_agents = data1['a'].unique()
            for agent_idx in uniq_agents[::per_agent]:
                data = data1[data1['a'] == agent_idx][:num_data]
                data = np.concatenate([np.array(data), np.expand_dims(np.array([scene_name] * len(data)),1)], axis=1)
                all_data.append(data)
        all_data = pd.DataFrame(np.concatenate(all_data))
        all_data.columns = ['frame', 'agent', 'pos_x', 'pos_y', 'scene']
        all_data.to_pickle(os.path.join(data_dir, data_split + '.pkl'))
        print(all_data.shape)

        # with open(save_path, 'wb') as handle:
        #     pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':

    traj = TrajectoryDataset(
            data_dir='../../datasets/Trajectories/Trajectories',
            data_split='val',
            skip=3)