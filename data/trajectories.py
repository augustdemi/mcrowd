import logging
import os
import math
import pandas as pd

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import derivative_of
from scipy import ndimage

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import imageio
import json

logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list,
     obs_frames, fut_frames, map_path, inv_h_t, local_map) = zip(*data)

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
    local_map = torch.cat(local_map, 0).permute((1, 0, 2, 3, 4))


    out = [
        obs_traj, pred_traj, seq_start_end,
        obs_frames, fut_frames, map_path, inv_h_t, local_map
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

        n_pred_state=2
        n_state=6
        # root_dir = '/dresden/users/ml1323/crowd/baseline/HTP-benchmark/A2E Data'
        # root_dir = 'C:\dataset\HTP-benchmark\A2E Data'
        root_dir = 'D:\crowd\datasets\Trajectories\Trajectories'

        all_files = [e for e in os.listdir(root_dir) if ('.csv' in e) and ('homo' not in e)]
        all_files = all_files[:2]
        with open(os.path.join(root_dir, 'exit_wc.json')) as data_file:
            all_exit_wc = json.load(data_file)

        # with open(self.data_dir) as f:
        #     all_files = np.array(f.readlines())
        # if 'Train' in self.data_dir:
        #     path_finding_files = all_files[['Pathfinding' in e for e in all_files]]
        #     all_files = np.concatenate((all_files[['Pathfinding' not in e for e in all_files]], np.repeat(path_finding_files, 10)))



        num_peds_in_seq = []
        seq_list = []

        obs_frame_num = []
        fut_frame_num = []
        map_file_names=[]
        inv_h_ts=[]


        for path in all_files:
            exit_wc = np.array(all_exit_wc[path])

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

            # data1 = data1[data1['a']<10]

            for agent_idx in data1['a'].unique():
                data = data1[data1['a'] == agent_idx][:50]
                # data = data1[data1['a'] == agent_idx].iloc[::3]
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
                    if len(frame_data[idx:idx + self.seq_len]) ==0:
                        print(idx)

                    curr_seq_data = np.concatenate(
                        frame_data[idx:idx + self.seq_len], axis=0) # frame을 seq_len만큼씩 잘라서 볼것 = curr_seq_data. 각 frame이 가진 데이터(agent)수는 다를수 잇음. 하지만 각 데이터의 길이는 4(frame #, agent id, pos_x, pos_y)
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1]) # unique agent id

                    curr_seq = np.zeros((len(peds_in_curr_seq), n_state, self.seq_len))
                    curr_seq_goals = np.zeros((len(peds_in_curr_seq), 2, 2)) # x, y position of exit and long-term goal
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

                        ## short term goal
                        pos = np.stack([x, y]).transpose((1, 0))
                        last_obs = pos[self.obs_len-1]
                        future_steps = pos[self.obs_len:]
                        long_goal= pos[-1]

                        dist_last2goal = np.linalg.norm(last_obs - long_goal)
                        candidates = np.where((exit_wc[:, 0] < last_obs[0] + dist_last2goal)
                                              & (exit_wc[:, 1] < last_obs[1] + dist_last2goal)
                                              & (last_obs[0] - dist_last2goal < exit_wc[:, 0])
                                              & (last_obs[1] - dist_last2goal < exit_wc[:, 1]))
                        candidates = exit_wc[candidates]

                        # 선택된 candidate중에 any of future steps와 가까운게 있는지 확인(exit을 안지날수 있으니)
                        short_goal = None
                        for c in candidates:
                            if any(((future_steps - c) ** 2).sum(1) < 1.5):
                                short_goal = c
                        if short_goal is None:  # short goal이 없다면 바로 long goal로
                            short_goal = long_goal
                        elif (
                            ((short_goal - last_obs) ** 2).sum() > ((short_goal - pos[6, :2]) ** 2).sum()) or \
                                (((short_goal - last_obs) ** 2).sum() > ((long_goal - pos[6, :2]) ** 2).sum()):
                            # short goal을 이미 과거 스텝에서 살짝 지나 온 경우 or long goal 보다 살짝 더 먼곳에 있는경우
                            short_goal = long_goal

                        # global_map = imageio.imread(map_file_name)
                        # plt.imshow(global_map)
                        # plt.scatter(pos[:, 0], pos[:20, 1], s=1, c='b')
                        # plt.scatter(pos[[7,19], 0], pos[[7,19], 1], s=10, c='b', marker='x')
                        # plt.scatter(short_goal[0], short_goal[1], s=20, c='green', marker='^')

                        curr_seq_goals[_idx, :] = np.stack([short_goal, long_goal])
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
            print(len(seq_list))
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
        for last_obs in self.obs_traj[start:end, :2, -1]:
            local_maps.append(get_local_map(last_obs, global_map, inv_h_t))
        local_maps = torch.stack(local_maps)

        # last_obs_wc -> last_obs_ic(1) -> 자름 -> last_obs_ic_on_local(2) : (1)과 (2)사이 x,y 이동값 계산
        # pred_ic_on_local -> pred_ic로 위의 x,y 이동 적용 -> pred_wc

        plt.scatter(82,18, s=1, c='r')
        plt.show()

        ##########
        obs_traj = self.obs_traj[start:end, :2][0].transpose(1,0)
        fut_traj = self.pred_traj[start:end, :2][0].transpose(1,0)
        plt.imshow(global_map)
        # plt.scatter(obs_traj[:,0], obs_traj[:,1], s=1, c='b')
        # plt.scatter(fut_traj[:,0], fut_traj[:,1], s=1, c='r')

        obs_pixel = np.round(obs_traj.numpy()).astype(int)
        fut_pixel = np.round(fut_traj.numpy()).astype(int)
        plt.scatter(obs_pixel[:,0], obs_pixel[:,1], s=1, c='b')
        plt.scatter(fut_pixel[:,0], fut_pixel[:,1], s=1, c='r')


        radius=24
        context_size = radius * 2
        expanded_obs_img = np.full(
            (global_map.shape[0] + context_size, global_map.shape[1] + context_size),
            False, dtype=np.float32)
        expanded_obs_img[context_size // 2:-context_size // 2,
        context_size // 2:-context_size // 2] = global_map.astype(np.float32)  # 99~-99

        target_pixel = [last_obs[1], last_obs[0]]
        img_pts = context_size // 2 + np.round(target_pixel).astype(int)

        local_map = expanded_obs_img[img_pts[0] - radius: img_pts[0] + radius,
                    img_pts[1] - radius: img_pts[1] + radius]

        local_map = transforms.Compose([
            transforms.Resize(112),
            transforms.ToTensor()
        ])(Image.fromarray(local_map))


        img_pts = context_size // 2 + np.round(obs_traj.numpy()).astype(int)
        plt.imshow(expanded_obs_img)
        plt.scatter(img_pts[:,0], img_pts[:,1], s=1, c='b')
        plt.show()


        local_map[radius-1, radius-1] = 206
        plt.imshow(local_map)
        plt.show()


        out = [
            self.obs_traj[start:end, :].to(self.device) , self.pred_traj[start:end, :].to(self.device),
            self.obs_frame_num[start:end], self.fut_frame_num[start:end],
            np.array([self.map_file_name[index]] * (end - start)), np.array([inv_h_t] * (end - start)), local_maps.to(self.device)
        ]
        return out