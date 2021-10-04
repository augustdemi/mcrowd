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
import pickle5

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
        self.data_path = os.path.join(data_dir, 'sdd_' + data_split + '.pkl')
        dt=0.4
        min_ped=0

        self.seq_len = self.obs_len + self.pred_len


        n_state = 6

        # all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []

        obs_frame_num = []
        fut_frame_num = []
        scene_names = []

        self.maps={}
        for file in os.listdir(self.map_dir):
            self.maps.update({file.split('.')[0]:imageio.imread(os.path.join(self.map_dir, file))})

        with open(self.data_path, 'rb') as f:
            data = pickle5.load(f)

        data = pd.DataFrame(data)
        scenes = data['sceneId'].unique()
        for s in scenes:
            # if (data_split=='train') and ('hyang_7' not in s):
            #     continue
            print(s)
            scene_data = data[data['sceneId'] == s]
            scene_data = scene_data.sort_values(by=['frame', 'trackId'], inplace=False)

            # print(scene_data.shape[0])
            frames = scene_data['frame'].unique().tolist()
            scene_data = np.array(scene_data)
            map_size = self.maps[s + '_mask'].shape
            scene_data[:,2] = np.clip(scene_data[:,2], a_min=None, a_max=map_size[1]-1)
            scene_data[:,3] =  np.clip(scene_data[:,2], a_min=None, a_max=map_size[0]-1)
            '''
            scene_data = data[data['sceneId'] == s]
            all_traj = np.array(scene_data)[:,2:4]
            # all_traj = np.array(scene_data[scene_data['trackId']==128])[:,2:4]
            plt.imshow(self.maps[s + '_mask'])
            plt.scatter(all_traj[:, 0], all_traj[:, 1], s=1, c='r')
            '''

            # print('uniq frames: ', len(frames))
            frame_data = []  # all data per frame
            for frame in frames:
                frame_data.append(scene_data[scene_data[:, 0]==frame])
                # frame_data.append(scene_data[scene_data['frame'] == frame])

            num_sequences = int(math.ceil((len(
                frames) - self.seq_len + 1) / self.skip))  # seq_len=obs+pred길이씩 잘라서 (input=obs, output=pred)주면서 train시킬것. 그래서 seq_len씩 slide시키면서 총 num_seq만큼의 iteration하게됨

            this_scene_seq = []

            # all frames를 seq_len(kernel size)만큼씩 sliding해가며 볼것. 이때 skip = stride.
            for idx in range(0, num_sequences * self.skip + 1, self.skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len],
                    axis=0)  # frame을 seq_len만큼씩 잘라서 볼것 = curr_seq_data. 각 frame이 가진 데이터(agent)수는 다를수 잇음. 하지만 각 데이터의 길이는 4(frame #, agent id, pos_x, pos_y)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])  # unique agent id

                curr_seq = np.zeros((len(peds_in_curr_seq), n_state, self.seq_len))
                num_peds_considered = 0
                ped_ids = []
                for _, ped_id in enumerate(peds_in_curr_seq):  # current frame sliding에 들어온 각 agent에 대해
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]  # frame#, agent id, pos_x, pos_y
                    # curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx  # sliding idx를 빼주는 이유?. sliding이 움직여온 step인 idx를 빼줘야 pad_front=0 이됨. 0보다 큰 pad_front라는 것은 현ped_id가 처음 나타난 frame이 desired first frame보다 더 늦은 경우.
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1  # pad_end까지선택하는 index로 쓰일거라 1더함
                    if (pad_end - pad_front != self.seq_len) or (curr_ped_seq.shape[0] != self.seq_len):  # seq_len만큼의 sliding동안 매 프레임마다 agent가 존재하지 않은 데이터였던것.
                        # print(curr_ped_seq.shape[0])
                        continue
                    ped_ids.append(ped_id)
                    # x,y,x',y',x'',y''
                    x = curr_ped_seq[:, 2].astype(float)
                    y = curr_ped_seq[:, 3].astype(float)
                    vx = derivative_of(x, dt)
                    vy = derivative_of(y, dt)
                    ax = derivative_of(vx, dt)
                    ay = derivative_of(vy, dt)

                    # Make coordinates relative
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = np.stack([x, y, vx, vy, ax, ay])
                    num_peds_considered += 1

                if num_peds_considered > min_ped:  # 주어진 하나의 sliding(16초)동안 등장한 agent수가 min_ped보다 큼을 만족하는 경우에만 이 slide데이터를 채택
                    num_peds_in_seq.append(num_peds_considered)
                    # 다음 list의 initialize는 peds_in_curr_seq만큼 해뒀었지만, 조건을 만족하는 slide의 agent만 차례로 append 되었기 때문에 num_peds_considered만큼만 잘라서 씀
                    seq_list.append(curr_seq[:num_peds_considered])
                    this_scene_seq.append(curr_seq[:num_peds_considered, :2])
                    obs_frame_num.append(np.ones((num_peds_considered, self.obs_len)) * frames[idx:idx + self.obs_len])
                    fut_frame_num.append(
                        np.ones((num_peds_considered, self.pred_len)) * frames[idx + self.obs_len:idx + self.seq_len])
                    scene_names.append([s] * num_peds_considered)
                    # inv_h_ts.append(inv_h_t)
                if data_split == 'test' and np.concatenate(this_scene_seq).shape[0] > 10:
                    break

        seq_list = np.concatenate(seq_list, axis=0) # (32686, 2, 16)
        self.obs_frame_num = np.concatenate(obs_frame_num, axis=0)
        self.fut_frame_num = np.concatenate(fut_frame_num, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)


        # frame seq순, 그리고 agent id순으로 쌓아온 데이터에 대한 index를 부여하기 위해 cumsum으로 index생성 ==> 한 슬라이드(16 seq. of frames)에서 고려된 agent의 data를 start, end로 끊어내서 index로 골래내기 위해
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist() # num_peds_in_seq = 각 slide(16개 frames)별로 고려된 agent수.따라서 len(num_peds_in_seq) = slide 수 = 2692 = self.num_seq
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ] # [(0, 2),  (2, 4),  (4, 7),  (7, 10), ... (32682, 32684),  (32684, 32686)]

        self.map_file_name = np.concatenate(scene_names)
        self.num_seq = len(self.obs_traj) # = slide (seq. of 16 frames) 수 = 2692

        print(self.seq_start_end[-1])


    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        global_map = np.expand_dims(self.maps[self.map_file_name[index] + '_mask'], axis=0)
        inv_h_t = np.expand_dims(np.eye(3), axis=0)
        local_ics = torch.cat([self.obs_traj[index, :2, :],  self.pred_traj[index, :2, :]], dim=1)[[1,0],:].detach().cpu().numpy()
        local_ics = np.round(local_ics).astype(int).transpose((1,0))

        #########
        out = [
            self.obs_traj[index].to(self.device).unsqueeze(0), self.pred_traj[index].to(self.device).unsqueeze(0),
            self.obs_frame_num[index], self.fut_frame_num[index],
            global_map, inv_h_t,
            global_map, np.expand_dims(local_ics, axis=0), inv_h_t
        ]
        return out
