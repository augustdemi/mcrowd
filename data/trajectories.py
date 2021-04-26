import logging
import os
import math
import pandas as pd

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import derivative_of

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import imageio

logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     obs_frames, fut_frames, past_obst, fut_obst) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    seq_start_end = torch.LongTensor(seq_start_end)

    obs_frames = np.concatenate(obs_frames, 0)
    fut_frames = np.concatenate(fut_frames, 0)

    past_obst = torch.cat(past_obst, 0).permute((1, 0, 2, 3, 4))
    fut_obst = torch.cat(fut_obst, 0).permute((1, 0, 2, 3, 4))


    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, seq_start_end, obs_frames, fut_frames, past_obst, fut_obst
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



def crop(map, target_pos1, inv_h_t, context_size=198):
    # map = imageio.imread(os.path.join('../datasets/syn_x/map/', 's2_map.png'))
    # h = np.loadtxt(os.path.join('../datasets/syn_x/map/','s2_H.txt'))
    # inv_h_t = np.linalg.pinv(np.transpose(h))

    nearby_area = context_size//2
    expanded_obs_img = np.full((map.shape[0] + context_size, map.shape[1] + context_size), False, dtype=np.float32)
    expanded_obs_img[context_size//2:-context_size//2, context_size//2:-context_size//2] = map.astype(np.float32) # 99~-99

    target_pos = np.expand_dims(target_pos1, 0)
    target_pixel = np.matmul(np.concatenate([target_pos, np.ones((len(target_pos), 1))], axis=1), inv_h_t)
    target_pixel /= np.expand_dims(target_pixel[:, 2], 1)
    target_pixel = target_pixel[:,:2]

    # plt.imshow(map)
    # plt.scatter(target_pixel[0][1], target_pixel[0][0], c='r', s=1)

    # img_pts = np.array([[289,106]]) # s5-160
    # img_pts = np.array([[272,63]]) # s5-64
    # img_pts = np.array([[263,38]]) # s5-64
    # img_pts = np.array([[246,38]]) # s5-16
    # img_pts = np.array([[279,62]]) #s2-64
    # img_pts = np.array([[265,44]]) #s2-32
    # img_pts = np.array([[264,38]]) #s2-16
    # img_pts = np.array([[273,62]]) #s3-64
    # img_pts = np.array([[257,40]]) #s3-16

    img_pts = context_size//2 + np.round(target_pixel).astype(int)

    # plt.imshow(expanded_obs_img)
    # plt.scatter(img_pts[0][1], img_pts[0][0], c='r', s=1)


    cropped_img = np.stack([expanded_obs_img[img_pts[i, 0] - nearby_area : img_pts[i, 0] + nearby_area,
                                      img_pts[i, 1] - nearby_area : img_pts[i, 1] + nearby_area]
                      for i in range(img_pts.shape[0])], axis=0)

    if (np.array(cropped_img.shape)[1:] < context_size).any():
        cropped_img = np.zeros((1, context_size, context_size)).astype('float32')


    cropped_img = np.kron(cropped_img, np.ones((4,4))).astype('float32')
    cropped_img[0, nearby_area*4, nearby_area*4] = 255

    # plt.imshow(cropped_img[0])
    return cropped_img


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1,
        min_ped=0, delim='\t', device='cpu', dt=0.4
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
        self.device = device
        self.map_dir = '../datasets/syn_x/map/'

        n_pred_state=2
        n_state=6

        self.context_size=16

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []

        obs_frame_num = []
        fut_frame_num = []
        map_file_names = []
        deli = '/'
        for path in all_files:
            print('data path:', path)

            map_file_name = path.split(deli)[-1]
            if 'bottleneck-evacuation_' in map_file_name:
                map_file_name = 's1'
            elif 'bottleneck-evacuation-2_' in map_file_name:
                map_file_name = 's2'
            elif 'squeeze_' in map_file_name:
                map_file_name = 's3'
            elif 'circles' in map_file_name:
                map_file_name = 's4'
            elif 'hallway-two-way_' in map_file_name:
                map_file_name = 's5'
            elif 'hallway-four-way_' in map_file_name:
                map_file_name = 's6'

            print('map path: ', map_file_name)

            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()


            # print('uniq frames: ', len(frames))
            frame_data = [] # all data per frame
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip)) # seq_len=obs+pred길이씩 잘라서 (input=obs, output=pred)주면서 train시킬것. 그래서 seq_len씩 slide시키면서 총 num_seq만큼의 iteration하게됨

            # all frames를 seq_len(kernel size)만큼씩 sliding해가며 볼것. 이때 skip = stride.
            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0) # frame을 seq_len만큼씩 잘라서 볼것 = curr_seq_data. 각 frame이 가진 데이터(agent)수는 다를수 잇음. 하지만 각 데이터의 길이는 4(frame #, agent id, pos_x, pos_y)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1]) # unique agent id

                curr_seq_rel = np.zeros((len(peds_in_curr_seq), n_pred_state, self.seq_len))
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
                    curr_seq[_idx, :, pad_front:pad_end] = np.stack([x, y, vx, vy, ax, ay])
                    curr_seq_rel[_idx, :, pad_front:pad_end] = np.stack([vx, vy])
                    num_peds_considered += 1


                if num_peds_considered > min_ped: # 주어진 하나의 sliding(16초)동안 등장한 agent수가 min_ped보다 큼을 만족하는 경우에만 이 slide데이터를 채택
                    num_peds_in_seq.append(num_peds_considered)
                    # 다음 list의 initialize는 peds_in_curr_seq만큼 해뒀었지만, 조건을 만족하는 slide의 agent만 차례로 append 되었기 때문에 num_peds_considered만큼만 잘라서 씀
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    obs_frame_num.append(np.ones((num_peds_considered, self.obs_len)) * frames[idx:idx + self.obs_len])
                    fut_frame_num.append(np.ones((num_peds_considered, self.pred_len)) * frames[idx + self.obs_len:idx + self.seq_len])
                    map_file_names.append(map_file_name)

            #     ped_ids = np.array(ped_ids)
            #     # if 'test' in path and len(ped_ids) > 0:
            #     if len(ped_ids) > 0:
            #         df.append([idx, len(ped_ids)])
            # df = np.array(df)
            # df = pd.DataFrame(df)
            # print(df.groupby(by=1).size())

            #     print("frame idx:", idx, "num_ped: ", len(ped_ids), " ped_ids: ", ",".join(ped_ids.astype(int).astype(str)))


        self.num_seq = len(seq_list) # = slide (seq. of 16 frames) 수 = 2692
        seq_list = np.concatenate(seq_list, axis=0) # (32686, 2, 16)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        self.obs_frame_num = np.concatenate(obs_frame_num, axis=0)
        self.fut_frame_num = np.concatenate(fut_frame_num, axis=0)


        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        # frame seq순, 그리고 agent id순으로 쌓아온 데이터에 대한 index를 부여하기 위해 cumsum으로 index생성 ==> 한 슬라이드(16 seq. of frames)에서 고려된 agent의 data를 start, end로 끊어내서 index로 골래내기 위해
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist() # num_peds_in_seq = 각 slide(16개 frames)별로 고려된 agent수.따라서 len(num_peds_in_seq) = slide 수 = 2692 = self.num_seq
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ] # [(0, 2),  (2, 4),  (4, 7),  (7, 10), ... (32682, 32684),  (32684, 32686)]
        self.map_file_name = map_file_names




    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        current_obs_traj = self.obs_traj[start:end, :].detach().clone()
        current_fut_traj = self.pred_traj[start:end, :].detach().clone()

        map_file_name = self.map_file_name[index]
        if map_file_name == 's4':
            map = np.zeros((500,500))
        else:
            map = imageio.imread(os.path.join(self.map_dir, map_file_name + '_map.png'))
        h = np.loadtxt(os.path.join(self.map_dir, map_file_name + '_H.txt'))


        inv_h_t = np.linalg.pinv(np.transpose(h))
        past_map_obst = []
        for i in range(end-start):  # len(past_obst) = batch
            seq_map = []
            for t in range(self.obs_len):
                cp_map = map.copy()
                cp_map = crop(cp_map, current_obs_traj[i,:2,t], inv_h_t, self.context_size) /255
                seq_map.append(cp_map)
            past_map_obst.append(np.stack(seq_map))
        past_map_obst = np.stack(past_map_obst) # (batch(start-end), 8, 1, map_size,map_size)
        past_map_obst = torch.from_numpy(past_map_obst)

        fut_map_obst = []
        for i in range(end-start):
            seq_map = []
            for t in range(self.pred_len):
                cp_map = map.copy()
                cp_map = crop(cp_map, current_fut_traj[i, :2, t], inv_h_t, self.context_size) /255
                seq_map.append(cp_map)
            fut_map_obst.append(np.stack(seq_map))
        fut_map_obst = np.stack(fut_map_obst)  # (batch(start-end), 12, 1, 128,128)
        fut_map_obst = torch.from_numpy(fut_map_obst)


        out = [
            self.obs_traj[start:end, :].to(self.device) , self.pred_traj[start:end, :].to(self.device),
            self.obs_traj_rel[start:end, :].to(self.device), self.pred_traj_rel[start:end, :].to(self.device),
            self.obs_frame_num[start:end], self.fut_frame_num[start:end],
            past_map_obst.to(self.device), fut_map_obst.to(self.device)
        ]
        return out
