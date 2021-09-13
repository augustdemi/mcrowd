import logging
import os
import math
import pandas as pd

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import derivative_of
from scipy import ndimage
import cv2

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import imageio
import json

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
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    seq_start_end = torch.LongTensor(seq_start_end)

    obs_frames = np.concatenate(obs_frames, 0)
    fut_frames = np.concatenate(fut_frames, 0)
    map_path = np.array(np.concatenate(map_path, 0))
    inv_h_t = np.array(np.concatenate(inv_h_t, 0))
    local_map = torch.cat(local_map, 0)
    local_ic = torch.cat(local_ic, 0)
    local_homo = torch.cat(local_homo, 0)


    out = [
        obs_traj, pred_traj, seq_start_end,
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


def get_local_map(last_obs_wc, global_map, inv_h_t, radius=12):
    context_size = radius * 2
    expanded_obs_img = np.full(
        (global_map.shape[0] + context_size, global_map.shape[1] + context_size),
        False, dtype=np.float32)
    expanded_obs_img[context_size // 2:-context_size // 2,
    context_size // 2:-context_size // 2] = global_map.astype(np.float32)  # 99~-99

    ######### TEST
    # plt.imshow(expanded_obs_img)
    # img_pts = context_size//2 + np.round(obs_pixel).astype(int)
    # for p in range(len(img_pts)):
    #     plt.scatter(img_pts[p][1], img_pts[p][0], c='b', s=1)
    #     expanded_obs_img[img_pts[p][0], img_pts[p][1]] = 0
    #
    # # fut_real = fut_traj
    # # fut_real = np.concatenate([fut_real, np.ones((12, 1))], axis=1)
    # # fut_pixel = np.matmul(fut_real, inv_h_t)
    # # fut_pixel /= np.expand_dims(fut_pixel[:, 2], 1)
    # # fut_pixel = fut_pixel[:, :2]
    # # fut_pixel[:, [1, 0]] = fut_pixel[:, [0, 1]]
    # img_pts = context_size//2 + np.round(fut_pixel).astype(int)
    # for p in range(len(img_pts)):
    #     plt.scatter(img_pts[p][1], img_pts[p][0], c='r', s=1)
    #     expanded_obs_img[img_pts[p][0], img_pts[p][1]] = 0
    # plt.show()
    ##############

    # target_pos = obs_real[-1]
    # target_pos[[0, 1]] = target_pos[[1, 0]]
    # target_pixel = np.matmul(
    #     np.concatenate([target_pos, (1,)], axis=0), inv_h_t)
    # target_pixel /= target_pixel[2]
    # target_pixel = target_pixel[:2]
    target_pixel = [last_obs_wc[1], last_obs_wc[0]]
    img_pts = context_size // 2 + np.round(target_pixel).astype(int)


    local_map = expanded_obs_img[img_pts[0] - radius: img_pts[0] + radius,
                img_pts[1] - radius: img_pts[1] + radius]
    # plt.imshow(local_map)
    local_map = transforms.Compose([
        transforms.ToTensor()
    ])(Image.fromarray(local_map))
    return local_map


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, context_size=198, resize=64,
        min_ped=0, delim='\t', device='cpu', dt=0.4, map_ae=False
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
        skip +=2
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.device = device
        self.context_size=context_size

        n_state=6
        root_dir = '/dresden/users/ml1323/crowd/datasets/Trajectories'
        # root_dir = 'D:\crowd\datasets\Trajectories\Trajectories'

        all_files = [e for e in os.listdir(root_dir) if ('.csv' in e) and ('homo' not in e)]
        all_files = np.array(sorted(all_files, key=lambda x: int(x.split('.')[0])))

        if data_dir.endswith('Train.txt'):
            all_files = all_files[:2]
            per_agent=5
            num_data=50
        elif data_dir.endswith('Val.txt'):
            all_files = all_files[[42,44]]
            per_agent=10
            num_data= 50
        else:
            all_files = all_files[[43,47,48,49]]
            per_agent=20
            num_data = 50

        # with open(os.path.join(root_dir, 'exit_wc.json')) as data_file:
        #     all_exit_wc = json.load(data_file)


        num_peds_in_seq = []
        seq_list = []

        obs_frame_num = []
        fut_frame_num = []
        map_file_names=[]
        inv_h_ts=[]


        for path in all_files:
            # exit_wc = np.array(all_exit_wc[path])

            path = os.path.join(root_dir, path.rstrip().replace('\\', '/'))
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
            # data = data1[data1['a'] < 10]
                frames = data['f'].unique().tolist()

                frame_data = []
                # data.sort_values(by=['f'])
                for frame in frames:
                    frame_data.append(data[data['f'] == frame].values)
                num_sequences = int(
                    math.ceil((len(frames) - self.seq_len + 1) / skip))
                # print('num_sequences: ', num_sequences)

                # all frames를 seq_len(kernel size)만큼씩 sliding해가며 볼것. 이때 skip = stride.
                for idx in range(0, num_sequences * self.skip + 1, skip):
                    # if len(frame_data[idx:idx + self.seq_len]) ==0:
                    #     print(idx)

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

                    if num_peds_considered > min_ped: # 주어진 하나의 sliding(16초)동안 등장한 agent수가 min_ped보다 큼을 만족하는 경우에만 이 slide데이터를 채택
                        num_peds_in_seq.append(num_peds_considered)
                        # 다음 list의 initialize는 peds_in_curr_seq만큼 해뒀었지만, 조건을 만족하는 slide의 agent만 차례로 append 되었기 때문에 num_peds_considered만큼만 잘라서 씀
                        seq_list.append(curr_seq[:num_peds_considered])
                        obs_frame_num.append(np.ones((num_peds_considered, self.obs_len)) * frames[idx:idx + self.obs_len])
                        fut_frame_num.append(np.ones((num_peds_considered, self.pred_len)) * frames[idx + self.obs_len:idx + self.seq_len])
                        # map_file_names.append(num_peds_considered*[map_file_name])
                        map_file_names.append(map_file_name)
                        inv_h_ts.append(inv_h_t)
            print(path, len(seq_list))
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
        self.map_file_name = map_file_names
        self.inv_h_t = inv_h_ts
        print(self.seq_start_end[-1])


    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        inv_h_t = self.inv_h_t[index]
        global_map = imageio.imread(self.map_file_name[index])

        local_maps =[]
        local_ics =[]
        local_homos =[]
        for idx in range(start, end):
            all_traj = torch.cat([self.obs_traj[idx, :2], self.pred_traj[idx, :2]], dim=1).transpose(1, 0).detach().cpu().numpy()
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
        local_maps = torch.stack(local_maps)
        local_ics = torch.stack(local_ics)
        local_homos = torch.stack(local_homos)


        ##########


        out = [
            self.obs_traj[start:end, :].to(self.device) , self.pred_traj[start:end, :].to(self.device),
            self.obs_frame_num[start:end], self.fut_frame_num[start:end],
            np.array([self.map_file_name[index]] * (end - start)), np.array([inv_h_t] * (end - start)),
            local_maps.to(self.device), local_ics.to(self.device), local_homos.to(self.device)
        ]
        return out

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


    local_map = transforms.Compose([
        transforms.ToTensor()
    ])(Image.fromarray(1-local_map/255))

    return local_map, torch.tensor(all_pixel_local), torch.tensor(h).float()
