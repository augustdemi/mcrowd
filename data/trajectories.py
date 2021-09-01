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

logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_frames, fut_frames,
     map_path, inv_h_t) = zip(*data)

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

    out = [
        obs_traj, pred_traj, seq_start_end,
        obs_frames, fut_frames, map_path, inv_h_t
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


def theta(v, w):
    return np.arccos(v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w)))



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
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.device = device
        n_pred_state=2
        n_state=6
        root_dir = '/dresden/users/ml1323/crowd/baseline/HTP-benchmark/A2E Data'
        # root_dir = 'C:\dataset\HTP-benchmark\A2E Data'

        self.context_size=context_size

        with open(self.data_dir) as f:
            all_files = f.readlines()
        num_peds_in_seq = []
        seq_list = []
        circle_list = []
        distance_list = []

        obs_frame_num = []
        fut_frame_num = []
        map_file_names=[]

        # stop = 0
        for path in all_files:
            # if stop > 0:
            #     break
            path = os.path.join(root_dir, path.rstrip().replace('\\', '/'))
            print('data path:', path)
            # if 'Pathfinding' not in path:
            #     continue
            # stop +=1
            map_file_name = path.replace('.txt', '.png')
            print('map path: ', map_file_name)
            h = np.loadtxt(map_file_name.replace('.png', '.hom'), delimiter=',')
            inv_h_t = np.linalg.pinv(np.transpose(h))
            map = ndimage.distance_transform_edt(imageio.imread(map_file_name) / 255)



            loaded_data = read_file(path, delim)

            data = pd.DataFrame(loaded_data)
            data.columns = ['f', 'a', 'pos_x', 'pos_y']
            data.sort_values(by=['f', 'a'], inplace=True)

            # data['a'] = 0
            # data = data.iloc[::3]
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
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0) # frame을 seq_len만큼씩 잘라서 볼것 = curr_seq_data. 각 frame이 가진 데이터(agent)수는 다를수 잇음. 하지만 각 데이터의 길이는 4(frame #, agent id, pos_x, pos_y)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1]) # unique agent id

                curr_seq = np.zeros((len(peds_in_curr_seq), n_state, self.seq_len))
                curr_seq_circle = np.zeros((len(peds_in_curr_seq), map.shape[0] , map.shape[1]))
                curr_seq_dist = np.zeros((len(peds_in_curr_seq), map.shape[0], map.shape[1]))
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
                    num_peds_considered += 1


                    # plt.imshow(map)
                    # fut_real = np.stack([x, y])[:,self.obs_len:].transpose((1,0))
                    # fut_real = np.concatenate([fut_real, np.ones((12, 1))], axis=1)
                    # fut_pixel = np.matmul(fut_real, inv_h_t)
                    # fut_pixel /= np.expand_dims(fut_pixel[:, 2], 1)
                    # fut_pixel = fut_pixel[:, :2]
                    # fut_pixel[:, [1, 0]] = fut_pixel[:, [0, 1]]
                    # plt.scatter(obs_pixel[:,1], obs_pixel[:,0], c='b', s=1)
                    # plt.scatter(fut_pixel[:, 1], fut_pixel[:, 0], s=1, c='r')
                    #
                    # plt.imshow(circle*distance)
                    # plt.scatter(obs_pixel[:,1], obs_pixel[:,0], c='b', s=1)
                    # plt.scatter(fut_pixel[:, 1], fut_pixel[:, 0], s=1, c='r')
                    #
                    # plt.scatter(obs_real[:,0], obs_real[:,1], c='b')
                    # plt.scatter(fut_real[:, 0], fut_real[:, 1], c='r')
                    # plt.scatter(goal_wc[:, 0], goal_wc[:, 1], c='g', marker='X')


                if num_peds_considered > min_ped: # 주어진 하나의 sliding(16초)동안 등장한 agent수가 min_ped보다 큼을 만족하는 경우에만 이 slide데이터를 채택
                    num_peds_in_seq.append(num_peds_considered)
                    # 다음 list의 initialize는 peds_in_curr_seq만큼 해뒀었지만, 조건을 만족하는 slide의 agent만 차례로 append 되었기 때문에 num_peds_considered만큼만 잘라서 씀
                    seq_list.append(curr_seq[:num_peds_considered])
                    circle_list.append(curr_seq_circle[:num_peds_considered])
                    distance_list.append(curr_seq_dist[:num_peds_considered])

                    obs_frame_num.append(np.ones((num_peds_considered, self.obs_len)) * frames[idx:idx + self.obs_len])
                    fut_frame_num.append(np.ones((num_peds_considered, self.pred_len)) * frames[idx + self.obs_len:idx + self.seq_len])
                    # map_file_names.append(num_peds_considered*[map_file_name])
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



    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        h=np.loadtxt(self.map_file_name[index].replace('.png', '.hom'), delimiter=',')
        inv_h_t = np.linalg.pinv(np.transpose(h))


        out = [
            self.obs_traj[start:end].to(self.device) , self.pred_traj[start:end].to(self.device),
            self.obs_frame_num[start:end], self.fut_frame_num[start:end],
            np.array([self.map_file_name[index]] * (end - start)) , np.array([inv_h_t] * (end - start))
        ]
        return out