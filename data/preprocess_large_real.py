import logging
import os
import math
import random

import pandas as pd
import cv2
import numpy as np
import torch
from toolz import curry
from torch.utils.data import Dataset

import sys
sys.path.append('../')
from utils import derivative_of

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import imageio
from skimage.transform import resize
import pickle5

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


def trajectory_curvature(t):
    path_distance = np.linalg.norm(t[-1] - t[0])

    lengths = np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1))  # Length between points
    path_length = np.sum(lengths)
    if np.isclose(path_distance, 0.):
        return 0, 0, 0
    return (path_length / path_distance) - 1


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_dir, data_split, device='cpu', scale=1, coll_th=0.2
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

        self.obs_len = 8
        self.pred_len = 12
        skip = 1
        self.scale = scale
        self.seq_len = self.obs_len + self.pred_len
        self.device = device
        delim = ','
        dt = 0.4
        min_ped = 0
        data_dir = data_dir.replace('\\', '/')

        n_state=6
        # root_dir = '/dresden/users/ml1323/crowd/datasets/Trajectories'
        # root_dir = os.path.join(data_dir, data_name)
        # root_dir = 'D:\crowd\datasets\Trajectories\Trajectories'

        if data_split == 'train':
            n=0
            n_sample = 4000
        elif data_split == 'val':
            n=1
            n_sample=500
        else:
            n=2
            n_sample=1000
        all_files = [e for e in os.listdir(data_dir) if ('.csv' in e) and ( (int(e.split('.csv')[0]) - n) % 10 == 0)]
        all_files = np.array(sorted(all_files, key=lambda x: int(x.split('.')[0])))
        # all_files = [all_files[-1]]


        num_peds_in_seq = []
        seq_list = []

        obs_frame_num = []
        fut_frame_num = []
        map_file_names=[]
        inv_h_ts=[]
        curvature = []

        for path in all_files:
            # exit_wc = np.array(all_exit_wc[path])
            num_data_from_one_file = 0
            path = os.path.join(data_dir, path.rstrip().replace('\\', '/'))
            print('data path:', path)
            # if 'Pathfinding' not in path:
            #     continue
            map_file_name = path.replace('.csv', '.png')
            print('map path: ', map_file_name)
            inv_h_t = np.eye(3)*2
            inv_h_t[-1, -1] = 1
            loaded_data = read_file(path, delim)

            data = pd.DataFrame(loaded_data[:,:4])
            data.columns = ['f', 'a', 'pos_x', 'pos_y']
            # data.sort_values(by=['f', 'a'], inplace=True)
            data.sort_values(by=['f', 'a'], inplace=True)

            frames = data['f'].unique().tolist()

            frame_data = []
            # data.sort_values(by=['f'])
            for frame in frames:
                frame_data.append(data[data['f'] == frame].values)
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))
            # print('num_sequences: ', num_sequences)

            # all frames를 seq_len(kernel size)만큼씩 sliding해가며 볼것. 이때 skip = stride.
            for idx in range(0, num_sequences * skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0) # frame을 seq_len만큼씩 잘라서 볼것 = curr_seq_data. 각 frame이 가진 데이터(agent)수는 다를수 잇음. 하지만 각 데이터의 길이는 4(frame #, agent id, pos_x, pos_y)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1]) # unique agent id

                curr_seq = np.zeros((len(peds_in_curr_seq), n_state, self.seq_len))
                num_peds_considered = 0
                ped_ids = []
                for _, ped_id in enumerate(peds_in_curr_seq): # current frame sliding에 들어온 각 agent에 대해
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :] # frame#, agent id, pos_x, pos_y
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx # sliding idx를 빼주는 이유?. sliding이 움직여온 step인 idx를 빼줘야 pad_front=0 이됨. 0보다 큰 pad_front라는 것은 현ped_id가 처음 나타난 frame이 desired first frame보다 더 늦은 경우.
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1 # pad_end까지선택하는 index로 쓰일거라 1더함
                    if pad_end - pad_front != self.seq_len: # seq_len만큼의 sliding동안 매 프레임마다 agent가 존재하지 않은 데이터였던것.
                        continue
                    ped_ids.append(ped_id)
                    # x,y,x',y',x'',y''
                    x = curr_ped_seq[:,2]
                    y = curr_ped_seq[:,3]
                    vx = derivative_of(x, dt)
                    vy = derivative_of(y, dt)
                    ax = derivative_of(vx, dt)
                    ay = derivative_of(vy, dt)

                    # Make coordinates relative
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = np.stack([x, y, vx, vy, ax, ay]) # (1,6,20)

                    num_peds_considered += 1
                if num_peds_considered > min_ped:  # 주어진 하나의 sliding(16초)동안 등장한 agent수가 min_ped보다 큼을 만족하는 경우에만 이 slide데이터를 채택
                    seq_traj = curr_seq[:num_peds_considered]

                    ## find the agent with max num of neighbors at the beginning of future steps
                    curr1 = seq_traj[:, :2, self.obs_len].repeat(num_peds_considered, 0)  # AAABBBCCC
                    curr2 = np.stack([seq_traj[:, :2, self.obs_len]] * num_peds_considered).reshape(-1, 2)  # ABCABC
                    dist = np.linalg.norm(curr1 - curr2, axis=1)
                    dist = dist.reshape(num_peds_considered, num_peds_considered)

                    if random.random() < 0.5:
                        d = random.randint(0, len(dist)-1)
                        target_agent_idx = np.where((dist[d] < 5))[0]
                    else:
                        target_agent_idx = []
                        for d in range(len(dist)):
                            neighbor_idx = np.where((dist[d] < 5))[0]
                            if len(neighbor_idx) > len(target_agent_idx):
                                target_agent_idx = neighbor_idx

                    seq_traj = seq_traj[target_agent_idx]
                    num_peds_considered = len(target_agent_idx)

                    for a in range(seq_traj.shape[0]):
                        gt_traj = seq_traj[a, :2].T
                        c = np.round(trajectory_curvature(gt_traj), 4)
                        curvature.append(c)
                        # if c > 100:
                        #     print(c)

                    '''
                    colors = ['red', 'magenta', 'lightgreen', 'slateblue', 'blue', 'darkgreen', 'darkorange',
                              'gray', 'purple', 'turquoise', 'midnightblue', 'olive', 'black', 'pink', 'burlywood',
                              'yellow']
                    global_map = imageio.imread(map_file_name)
                    env = np.stack([global_map, global_map, global_map]).transpose(1, 2, 0) / 255
                    plt.imshow(env)
                    
                    cc = []
                    for idx in range(seq_traj.shape[0]):
                        gt_xy = seq_traj[idx, :2].T
                        c = np.round(trajectory_curvature(gt_xy),4)
                        cc.append(c)
                        print(c, colors[idx%16])
                        all_traj = gt_xy * 2
                        plt.plot(all_traj[:, 0], all_traj[:, 1], c=colors[idx % 16], marker='.', linewidth=1)
                        plt.scatter(all_traj[0, 0], all_traj[0, 1], s=30, c=colors[idx % 16], marker='x')
                        # plt.scatter(all_traj[:, 0], all_traj[:, 1], c=colors[idx%16], s=1)
                        # plt.scatter(all_traj[0, 0], all_traj[0, 1], s=20, c=colors[idx%16], marker='x')
                    plt.show()
                    cc = np.array(cc)
                    n, bins, patches = plt.hist(cc)

                    '''

                    #######
                    # curr1 = seq_traj[:, :2, self.obs_len].repeat(num_peds_considered, 0)  # AAABBBCCC
                    # curr2 = np.stack([seq_traj[:, :2, self.obs_len]] * num_peds_considered).reshape(-1, 2)  # ABCABC
                    # dist = np.linalg.norm(curr1 - curr2, axis=1)
                    # dist = dist.reshape(num_peds_considered, num_peds_considered)
                    # diff_agent_idx = np.triu_indices(num_peds_considered, k=1)
                    # if len(dist[diff_agent_idx]) >0:
                    #     print(np.round(dist[diff_agent_idx].min(), 2), np.round(dist[diff_agent_idx].max(), 2))

                    #######

                    seq_list.append(seq_traj)
                    num_data_from_one_file += num_peds_considered
                    num_peds_in_seq.append(num_peds_considered)
                    obs_frame_num.append(np.ones((num_peds_considered, self.obs_len)) * frames[idx:idx + self.obs_len])
                    fut_frame_num.append(
                        np.ones((num_peds_considered, self.pred_len)) * frames[idx + self.obs_len:idx + self.seq_len])
                    # map_file_names.append(num_peds_considered*[map_file_name])
                    map_file_names.append(map_file_name)
                    inv_h_ts.append(inv_h_t)
                # if frames[idx + self.obs_len] >= 1840:
                #     break
                if num_data_from_one_file > n_sample:
                    break
            cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
            aa = np.array([(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])])
            print('num data,  min/avg/max #agent')
            print(num_data_from_one_file, np.round((aa[:, 1] - aa[:, 0]).min(),2), np.round((aa[:, 1] - aa[:, 0]).mean(), 2), np.round((aa[:, 1] - aa[:, 0]).max(), 2))

        seq_list = np.concatenate(seq_list, axis=0)  # (32686, 2, 16)
        self.obs_frame_num = np.concatenate(obs_frame_num, axis=0)
        self.fut_frame_num = np.concatenate(fut_frame_num, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = seq_list[:, :, :self.obs_len]
        self.fut_traj = seq_list[:, :, self.obs_len:]

        # frame seq순, 그리고 agent id순으로 쌓아온 데이터에 대한 index를 부여하기 위해 cumsum으로 index생성 ==> 한 슬라이드(16 seq. of frames)에서 고려된 agent의 data를 start, end로 끊어내서 index로 골래내기 위해
        cum_start_idx = [0] + np.cumsum(
            num_peds_in_seq).tolist()  # num_peds_in_seq = 각 slide(16개 frames)별로 고려된 agent수.따라서 len(num_peds_in_seq) = slide 수 = 2692 = self.num_seq
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]  # [(0, 2),  (2, 4),  (4, 7),  (7, 10), ... (32682, 32684),  (32684, 32686)]

        # self.num_seq = len(self.seq_start_end)
        self.num_seq = len(self.obs_traj)
        self.map_file_name = map_file_names
        self.inv_h_t = inv_h_ts
        self.local_map = []
        self.local_homo = []
        self.local_ic = []
        print(self.seq_start_end[-1])


        c = np.array(curvature)
        # n, bins, patches = plt.hist(c)
        # plt.show()
        np.save(data_split + '_curvature.npy', c)
        print(c.min(), np.round(c.mean(),4), np.round(c.max(),4))
        c.sort()
        print(np.round(c[len(c)//2]))


        for seq_i in range(len(self.seq_start_end)):
            start, end = self.seq_start_end[seq_i]
            global_map = imageio.imread(self.map_file_name[seq_i])

            local_maps =[]
            local_ics =[]
            local_homos =[]
            for idx in range(start, end):
                all_traj = np.concatenate([self.obs_traj[idx, :2], self.fut_traj[idx, :2]], axis=1).transpose(1, 0)
                # plt.imshow(global_map)
                # plt.scatter(all_traj[:8,0], all_traj[:8,1], s=1, c='b')
                # plt.scatter(all_traj[8:,0], all_traj[8:,1], s=1, c='r')
                # plt.show()
                local_map, local_ic, local_h = get_local_map_ic(global_map, all_traj, zoom=10, radius=8)
                local_maps.append(local_map)
                local_ics.append(local_ic)
                local_homos.append(local_h)
                # plt.imshow(local_map[0])
                # plt.scatter(local_ic[:,1], local_ic[:,0], s=1, c='r')
                # plt.show()
            obs_real = np.concatenate([self.obs_traj[idx, :2].transpose(1,0), np.ones((self.obs_len, 1))], axis=1)
            obs_pixel = np.matmul(obs_real, inv_h_t)
            obs_pixel /= np.expand_dims(obs_pixel[:, 2], 1)
            '''
            seq_i+=1
            start, end = self.seq_start_end[seq_i]
            global_map = imageio.imread(self.map_file_name[seq_i])

            env = np.stack([global_map, global_map, global_map]).transpose(1,2,0) / 255
            plt.imshow(env)
            colors = ['red', 'magenta', 'lightgreen', 'slateblue', 'blue', 'darkgreen', 'darkorange',
                 'gray', 'purple', 'turquoise', 'midnightblue', 'olive', 'black', 'pink', 'burlywood', 'yellow']

            for idx in range(start, end):
                all_traj = np.concatenate([self.obs_traj[idx, :2], self.fut_traj[idx, :2]], axis=1).transpose(1, 0) * 2
                plt.plot(all_traj[:, 0], all_traj[:, 1], c=colors[idx%16], marker='.', linewidth=1)
                plt.scatter(all_traj[0, 0], all_traj[0, 1], s=20, c=colors[idx%16], marker='x')
                # plt.scatter(all_traj[:, 0], all_traj[:, 1], c=colors[idx%16], s=1)
                # plt.scatter(all_traj[0, 0], all_traj[0, 1], s=10, c=colors[idx%16], marker='x')
            plt.show()
            '''

            self.local_map.append(np.stack(local_maps))
            self.local_ic.append(np.stack(local_ics))
            self.local_homo.append(np.stack(local_homos))
        self.local_map = np.concatenate(self.local_map)
        self.local_ic = np.concatenate(self.local_ic)
        self.local_homo = np.concatenate(self.local_homo)

        all_data = \
            {'seq_start_end': self.seq_start_end,
             'obs_traj': self.obs_traj,
             'fut_traj': self.fut_traj,
             'obs_frame_num': self.obs_frame_num,
             'fut_frame_num': self.fut_frame_num,
             'map_file_name': self.map_file_name,
             'inv_h_t': self.inv_h_t,
             'local_map': self.local_map,
             'local_ic': self.local_ic,
             'local_homo': self.local_homo,
             }

        save_path = os.path.join(data_dir, data_split + '.pkl')
        with open(save_path, 'wb') as handle:
            pickle5.dump(all_data, handle, protocol=pickle5.HIGHEST_PROTOCOL)






if __name__ == '__main__':
    path = '../../datasets/large_real/Trajectories'
    # path = 'C:\dataset\large-real/Trajectories'
    coll_th = 0.5
    traj = TrajectoryDataset(
            data_dir=path,
            data_split='test',
            device='cpu',
            scale=1,
            coll_th=coll_th)