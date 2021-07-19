import logging
import os
import math
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import derivative_of

import matplotlib.pyplot as plt
from torchvision import transforms
import imageio
import torchvision.transforms.functional as TF
import random
import pandas as pd
logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list,
     obs_frames, fut_frames, past_obst, fut_obst) = zip(*data)

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

    past_obst = torch.cat(past_obst, 0).permute((1, 0, 2, 3, 4))
    fut_obst = torch.cat(fut_obst, 0).permute((1, 0, 2, 3, 4))

    out = [
        obs_traj, pred_traj, seq_start_end, obs_frames, fut_frames, past_obst, fut_obst
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



def crop(map, target_pos, inv_h_t, context_size=198):
    # context_size=32
    expanded_obs_img = np.full((map.shape[0] + context_size, map.shape[1] + context_size), False, dtype=np.float32)
    expanded_obs_img[context_size//2:-context_size//2, context_size//2:-context_size//2] = map.astype(np.float32) # 99~-99

    # target_pos = np.expand_dims(target_poss, 0)
    target_pixel = np.matmul(np.concatenate([target_pos, np.ones((len(target_pos), 1))], axis=1), inv_h_t)
    target_pixel /= np.expand_dims(target_pixel[:, 2], 1)
    target_pixel = target_pixel[:,:2]

    # plt.imshow(map)
    # for i in range(len(target_pixel)):
    #     plt.scatter(target_pixel[i][0], target_pixel[i][1], c='r', s=1)
    # plt.show()

    ######## TEST #########
    # trajectories/17.txt
    # pts = [[90.18498,61.6496], [90.29415,65.84846], [90.31033,66.44829], [90.3265,67.04811], [90.34267,67.64793],[90.34,73.03415]]

    # trajectories/0.txt
    # pts = [[63.73397, 46.06736], [64.28874, 45.84676], [64.84896, 45.64536], [65.41486, 45.47386],
    #        [65.98319,45.29537], [66.55249,45.11207],[67.12218,44.92652],[67.69204,44.73989], [68.26196,44.55278], [73.98132,42.76708],[74.57302,42.68015]]
    # # zara1 data points
    # pts = [[4.1068, 7.5457], [3.90560103932, 7.51993162917], [3.70418592941, 7.493917711], [3.52529058623, 7.48795121601], [3.34639524305, 7.48198472102],  [3.16749989986, 7.47601822602],
    #        [2.98860455668,	7.47005173103], [2.95282548805,	7.41921719369], [2.91704641941, 7.36838265635], [2.98860455668, 7.47005173103], [2.95282548805, 7.41921719369]]
    # pts = [[3.71976034752, 3.09875883948], [3.26957547803, 3.12095420085], [2.81918014343, 3.14314956223], [2.36899527394,3.1653449236], [2.03098830789,	3.18754028497], [1.76790692085,	3.20997430615],
    #        [1.50482553382,	3.23240832732], [1.34087321342,	3.2483985339], [1.32571972553, 3.25484234849]]
    # pts = [[11.5907345173, 5.95718726065], [11.1112949976, 5.86434859856], [10.6356438498, 5.81518467982], [10.1688322367, 5.86721251616], [9.70181015841, 5.91947901229], [9.23436714993, 5.97890530242],
    #        [8.76250437415, 6.10229241887], [8.29085206348, 6.22567953533], [7.82109393879, 6.32734861]]
    # students001
    # pts = [[2.37867666899, 6.29345891844],[2.45044527137,6.19059654477], [2.52221387375, 6.08749551129], [2.59398247613,5.98439447781], [2.66575107852, 5.88129344434],
    #        [8.1995102059,1.41024075651], [8.66821600504, 1.10642683147], [9.0807276199, 0.84151445379], [9.43683458539,0.615026303862], [9.79315201599, 0.388538153933],
    #        [10.1492589815, 0.162050004005]]
    # pts = [[0.064612788655,	0.623618056651], [0.260976735936,0.591876303289],[0.457340683216,0.560134549927], [0.653915095607,0.528154136766],
    #        [6.86410908533, 0.270401553075], [7.34523232593, 0.190927839771], [7.66092999037, 0.0835309298997], [7.97662765481, -0.0238659799713]]
    # univ
    # pts = [[14.8422099959, 8.29175742144], [14.3461437325, 8.36001412416], [13.8500774691, 8.42803216708], [13.3464344618, 8.54736206694],
    #        [10.4272833913, 9.11107651386], [9.94573922046, 9.26190950728], [9.4742973749, 9.42419817108], [9.00348692467,9.58815745349], [8.53877996262,	9.76715230327],
    #        [1.39601507215, 10.4905301562], [0.84291276405, 10.5404100543], [0.289810455954,10.5902899525], [-0.263291852142, 10.6404085104]]
    #
    # target_pixel = np.matmul(np.concatenate([pts, np.ones((len(pts), 1))], axis=1), inv_h_t)
    # target_pixel /= np.expand_dims(target_pixel[:, 2], 1)
    # target_pixel = target_pixel[:,:2]
    # img_pts = context_size//2 + np.round(target_pixel).astype(int)
    #
    # plt.imshow(expanded_obs_img)
    # for p in range(len(img_pts)):
    #     plt.scatter(img_pts[p][0], img_pts[p][1], c='r', s=1)
    #     expanded_obs_img[img_pts[p][1], img_pts[p][0]] = 0
    # plt.show()
    #########
    # plt.imshow(expanded_obs_img)
    # plt.scatter(img_pts[0][1], img_pts[0][0], c='r', s=1)
    # expanded_obs_img[img_pts[0][0], img_pts[0][1]]=255

    img_pts = context_size//2 + np.round(target_pixel).astype(int)

    nearby_area = context_size//2
    cropped_img = np.stack([expanded_obs_img[img_pts[i, 1] - nearby_area : img_pts[i, 1] + nearby_area,
                                      img_pts[i, 0] - nearby_area : img_pts[i, 0] + nearby_area]
                      for i in range(target_pos.shape[0])], axis=0)

    cropped_img[:, nearby_area, nearby_area] = 0
    # plt.imshow(cropped_img[0])
    # plt.show()
    return cropped_img



class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, context_size=198,
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
        n_state=6
        root_dir = '/dresden/users/ml1323/crowd/baseline/HTP-benchmark/A2E Data'
        # root_dir = 'C:\dataset\HTP-benchmark\A2E Data'

        self.context_size=context_size

        with open(self.data_dir) as f:
            all_files = f.readlines()
        num_peds_in_seq = []
        seq_list = []

        obs_frame_num = []
        fut_frame_num = []
        map_file_names=[]
        for path in all_files:
            path = os.path.join(root_dir, path.rstrip().replace('\\', '/'))
            print('data path:', path)
            # if 'Pathfinding' not in path:
            #     continue
            map_file_name = path.replace('.txt', '.png')
            print('map path: ', map_file_name)

            loaded_data = read_file(path, delim)

            data = pd.DataFrame(loaded_data)
            data.columns = ['f', 'a', 'pos_x', 'pos_y']
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
            for idx in range(0, num_sequences * self.skip + 1, skip):
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
                    curr_seq[_idx, :, pad_front:pad_end] = np.stack([x, y, vx, vy, ax, ay])
                    num_peds_considered += 1

                if num_peds_considered > min_ped: # 주어진 하나의 sliding(16초)동안 등장한 agent수가 min_ped보다 큼을 만족하는 경우에만 이 slide데이터를 채택
                    num_peds_in_seq.append(num_peds_considered)
                    # 다음 list의 initialize는 peds_in_curr_seq만큼 해뒀었지만, 조건을 만족하는 slide의 agent만 차례로 append 되었기 때문에 num_peds_considered만큼만 잘라서 씀
                    seq_list.append(curr_seq[:num_peds_considered])
                    obs_frame_num.append(np.ones((num_peds_considered, self.obs_len)) * frames[idx:idx + self.obs_len])
                    fut_frame_num.append(np.ones((num_peds_considered, self.pred_len)) * frames[idx + self.obs_len:idx + self.seq_len])
                    # map_file_names.append(num_peds_considered*[map_file_name])
                    map_file_names.append(map_file_name)
            print(sum(num_peds_in_seq))

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
        current_obs_traj = self.obs_traj[start:end, :].detach().clone()
        current_fut_traj = self.pred_traj[start:end, :].detach().clone()
        map = imageio.imread(self.map_file_name[index])
        h=np.loadtxt(self.map_file_name[index].replace('.png', '.hom'), delimiter=',')

        inv_h_t = np.linalg.pinv(np.transpose(h))
        past_map_obst = []
        for i in range(end-start):  # len(past_obst) = batch
            seq_map = []
            cp_map = map.copy()
            cropped_maps = crop(cp_map, current_obs_traj[i, :2].transpose(1,0), inv_h_t, self.context_size)
            for t in range(self.obs_len):
                cropped_map = transforms.Compose([
                    # transforms.Resize(32),
                    transforms.ToTensor()
                ])(Image.fromarray(cropped_maps[t]))
                seq_map.append(cropped_map / 255.0)
            past_map_obst.append(np.stack(seq_map))
        past_map_obst = np.stack(past_map_obst) # (batch(start-end), 8, 1, map_size,map_size)
        # past_map_obst[:,:,:,self.context_size, self.context_size] = 0
        past_map_obst = torch.from_numpy(past_map_obst)

        fut_map_obst = []
        for i in range(end-start):
            seq_map = []
            cp_map = map.copy()
            cropped_maps = crop(cp_map, current_fut_traj[i, :2].transpose(1, 0), inv_h_t, self.context_size)
            for t in range(self.pred_len):
                cropped_map = transforms.Compose([
                    # transforms.Resize(32),
                    transforms.ToTensor()
                ])(Image.fromarray(cropped_maps[t]))
                seq_map.append(cropped_map / 255.0)
            fut_map_obst.append(np.stack(seq_map))
        fut_map_obst = np.stack(fut_map_obst)  # (batch(start-end), 12, 1, 128,128)
        # fut_map_obst[:,:,:,self.context_size//2, self.context_size//2] = 0
        fut_map_obst = torch.from_numpy(fut_map_obst)


        out = [
            current_obs_traj.to(self.device), current_fut_traj.to(self.device),
            self.obs_frame_num[start:end], self.fut_frame_num[start:end],
            past_map_obst.to(self.device), fut_map_obst.to(self.device)
        ]
        return out
