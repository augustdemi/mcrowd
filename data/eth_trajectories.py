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

logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list,
     obs_frames, fut_frames, map_path, inv_h_t,
     local_map, local_ic, local_homo, local_map_size) = zip(*data)

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
    map_path = np.concatenate(map_path, 0)
    inv_h_t = np.concatenate(inv_h_t, 0)
    # local_map = np.array(np.concatenate(local_map, 0))
    local_map = torch.cat(local_map, 0)
    local_ic = np.concatenate(local_ic, 0)
    local_homo = torch.cat(local_homo, 0)
    local_map_size = np.concatenate(local_map_size, 0)


    out = [
        obs_traj, pred_traj, seq_start_end,
        obs_frames, fut_frames, map_path, inv_h_t,
        local_map, local_ic, local_homo, local_map_size
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
        # self.data_dir = '../../datasets/eth/test'
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = 1
        self.seq_len = self.obs_len + self.pred_len
        self.delim = '\t'
        self.device = device
        self.map_dir =  '../datasets/nmap/map/'

        n_pred_state=2
        n_state=6

        self.context_size=context_size
        self.resize=resize

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []

        seq_past_obst_list = []
        seq_fut_obst_list = []
        obs_frame_num = []
        fut_frame_num = []
        map_file_names=[]
        local_map_size = []
        deli = '/'
        # deli = '\\'

        for path in all_files:
            print('data path:', path)
            # if 'zara' in path or 'eth' in path or 'hotel' in path:
            # if 'zara' in path or 'hotel' in path:
            #     continue
            # if 'students003' in path:
            #     continue
            if 'zara01' in path.split(deli)[-1]:
                map_file_name = 'zara01'
            elif 'zara02' in path.split(deli)[-1]:
                map_file_name = 'zara02'
            elif 'eth' in path.split(deli)[-1]:
                map_file_name = 'eth'
            elif 'hotel' in path.split(deli)[-1]:
                map_file_name = 'hotel'
            elif 'students003' in path.split(deli)[-1]:
                map_file_name = 'univ'
            else:
                if skip > 0:
                    map_file_name = 'skip'
                    print('map path: ', map_file_name)

                    continue
                else:
                    map_file_name = ''

            print('map path: ', map_file_name)


            data = read_file(path, self.delim)
            # print('uniq ped: ', len(np.unique(data[:, 1])))


            if 'zara01' in map_file_name:
                frames = (np.unique(data[:, 0]) + 10).tolist()
            else:
                frames = np.unique(data[:, 0]).tolist()

            if 'test' in all_files:
                frames = frames[:(len(frames)//3)*2]

            # print('uniq frames: ', len(frames))
            frame_data = [] # all data per frame
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / self.skip)) # seq_len=obs+pred길이씩 잘라서 (input=obs, output=pred)주면서 train시킬것. 그래서 seq_len씩 slide시키면서 총 num_seq만큼의 iteration하게됨

            this_seq_obs = []

            # all frames를 seq_len(kernel size)만큼씩 sliding해가며 볼것. 이때 skip = stride.
            for idx in range(0, num_sequences * self.skip + 1, self.skip):
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

                    ### others
                    per_frame_past_obst = []
                    per_frame_fut_obst = []
                    if map_file_name is '':
                        per_frame_past_obst = [[]] * self.obs_len
                        per_frame_fut_obst = [[]] * self.pred_len
                    else:
                        curr_obst_seq = curr_seq_data[curr_seq_data[:, 1] != ped_id, :] # frame#, agent id, pos_x, pos_y
                        i=0
                        for frame in np.unique(curr_ped_seq[:,0]): # curr_ped_seq는 continue를 지나왔으므로 반드시 20임
                            neighbor_ped = curr_obst_seq[curr_obst_seq[:, 0] == frame][:, 2:]
                            if i < self.obs_len:
                                # print('neighbor_ped:', len(neighbor_ped))
                                if len(neighbor_ped) ==0:
                                    per_frame_past_obst.append([])
                                else:
                                    per_frame_past_obst.append(np.around(neighbor_ped, decimals=4))
                            else:
                                if len(neighbor_ped) ==0:
                                    per_frame_fut_obst.append([])
                                else:
                                    per_frame_fut_obst.append(np.around(neighbor_ped, decimals=4))
                            i += 1
                    seq_past_obst_list.append(per_frame_past_obst)
                    seq_fut_obst_list.append(per_frame_fut_obst)



                if num_peds_considered > min_ped: # 주어진 하나의 sliding(16초)동안 등장한 agent수가 min_ped보다 큼을 만족하는 경우에만 이 slide데이터를 채택
                    num_peds_in_seq.append(num_peds_considered)
                    # 다음 list의 initialize는 peds_in_curr_seq만큼 해뒀었지만, 조건을 만족하는 slide의 agent만 차례로 append 되었기 때문에 num_peds_considered만큼만 잘라서 씀
                    seq_list.append(curr_seq[:num_peds_considered])
                    this_seq_obs.append(curr_seq[:num_peds_considered, :2, :8])
                    obs_frame_num.append(np.ones((num_peds_considered, self.obs_len)) * frames[idx:idx + self.obs_len])
                    fut_frame_num.append(np.ones((num_peds_considered, self.pred_len)) * frames[idx + self.obs_len:idx + self.seq_len])
                    # map_file_names.append(num_peds_considered*[map_file_name])
                    map_file_names.append(map_file_name)

            if map_file_name == '':
                per_step_dist = []
                for obs_traj in np.concatenate(this_seq_obs):
                    obs_traj = obs_traj.transpose(1, 0)
                    per_step_dist.append(np.sqrt(((obs_traj[1:] - obs_traj[:-1]) ** 2).sum(1)).mean())
                mean_pixel_dist = 0.7
                # argmax = np.concatenate(this_seq_obs)[np.array(per_step_dist).argmax()].transpose(1,0)
                # plt.scatter(argmax[:, 1], argmax[:, 0], s=1)

            else:
                h = np.loadtxt(os.path.join(self.map_dir, map_file_name + '_H.txt'))
                inv_h_t = np.linalg.pinv(np.transpose(h))
                all_pixels = []
                for obs_traj in np.concatenate(this_seq_obs):
                    obs_traj = obs_traj.transpose(1, 0)
                    all_pixel = np.matmul(np.concatenate([obs_traj, np.ones((len(obs_traj), 1))], axis=1),
                                          inv_h_t)
                    all_pixel /= np.expand_dims(all_pixel[:, 2], 1)
                    all_pixels.append(all_pixel[:, :2])
                all_pixels = np.stack(all_pixels)
                per_step_dist = []
                for all_pixel in all_pixels:
                    per_step_dist.append(np.sqrt(((all_pixel[1:] - all_pixel[:-1]) ** 2).sum(1)).mean())

                two_wc_pts = np.array([[0,0], [0,0.7]])
                two_ic_pts = np.matmul(np.concatenate([two_wc_pts, np.ones((len(two_wc_pts), 1))], axis=1),
                                      inv_h_t)
                two_ic_pts /= np.expand_dims(two_ic_pts[:, 2], 1)
                mean_pixel_dist = np.linalg.norm(two_ic_pts[1,:2] - two_ic_pts[0,:2])


                # map_path = os.path.join(self.map_dir, map_file_name + '_map.png')
                # global_map = imageio.imread(map_path)
                # plt.imshow(global_map)
                # argmax = all_pixels[np.array(per_step_dist).argmax()]
                # plt.scatter(argmax[:, 1], argmax[:, 0], s=1)

            per_step_dist = np.array(per_step_dist)
            # max_per_step_dist_of_seq = per_step_dist[np.where(per_step_dist>0.1)[0]].max()
            # max_per_step_dist_of_seq = per_step_dist.max()
            # local_map_size.extend([int(max_per_step_dist_of_seq * 13)] * len(this_seq_obs))
            local_map_size.extend(list((np.clip(per_step_dist, mean_pixel_dist, None) * 18).astype(int)))




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
        self.past_obst = seq_past_obst_list
        self.fut_obst = seq_fut_obst_list

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
        self.local_map_size = local_map_size

        print(self.seq_start_end[-1])


    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        local_maps = []
        local_ics = []
        local_homos = []
        if self.map_file_name[index] == '':
            zoom=40
            inv_h_t = np.zeros((3,3))
            for idx in range(start, end):
                all_traj = torch.cat([self.obs_traj[idx, :2], self.pred_traj[idx, :2]], dim=1).\
                    transpose(1, 0).detach().cpu().numpy()
                local_map, local_ic, local_h = get_local_map_ic_no_map(all_traj, zoom=zoom, radius=self.local_map_size[idx])
                local_maps.append(local_map)
                local_ics.append(local_ic)
                local_homos.append(local_h)
            local_map_size = np.array(self.local_map_size[start:end]) * 2 * zoom
        else:
            map_path = os.path.join(self.map_dir, self.map_file_name[index] + '_map.png')
            global_map = 255 - imageio.imread(map_path)
            h = np.loadtxt(os.path.join(self.map_dir, self.map_file_name[index] + '_H.txt'))
            inv_h_t = np.linalg.pinv(np.transpose(h))

            for idx in range(start, end):
                all_traj = torch.cat([self.obs_traj[idx, :2], self.pred_traj[idx, :2]], dim=1).\
                    transpose(1, 0).detach().cpu().numpy()
                # plt.imshow(global_map)
                # plt.scatter(all_traj[:8,0], all_traj[:8,1], s=1, c='b')
                # plt.scatter(all_traj[8:,0], all_traj[8:,1], s=1, c='r')
                # plt.show()
                #eth = 256, zara1 =384 = hotel
                # students003: 470
                local_map, local_ic, local_h = get_local_map_ic(global_map, all_traj, inv_h_t, zoom=1, radius=self.local_map_size[idx])
                local_maps.append(local_map)
                local_ics.append(local_ic)
                local_homos.append(local_h)
            local_map_size = np.array(self.local_map_size[start:end]) * 2

                # plt.imshow(local_map[0])
            # plt.scatter(local_ic[:, 1], local_ic[:, 0], s=1)
            # plt.scatter(local_ic[7, 1], local_ic[7, 0], s=1, c='r')
        local_maps = torch.stack(local_maps)
        local_ics = np.stack(local_ics)
        local_homos = torch.stack(local_homos)

        #########
        out = [
            self.obs_traj[start:end, :].to(self.device), self.pred_traj[start:end, :].to(self.device),
            self.obs_frame_num[start:end], self.fut_frame_num[start:end],
            np.array([self.map_file_name[index]] * (end - start)), np.array([inv_h_t] * (end - start)),
            local_maps.to(self.device), local_ics, local_homos.to(self.device), local_map_size
        ]
        return out

def get_local_map_ic_no_map(all_traj, zoom=20, radius=8):
    radius = radius * zoom
    context_size = radius * 2

    all_pixel = all_traj[:, [1, 0]] * zoom
    dist = radius - all_pixel[7]
    local_ic = np.round(all_pixel + dist).astype(int)

    local_map = np.zeros((context_size, context_size))

    # plt.imshow(local_map)
    # plt.scatter(local_ic[:, 1], local_ic[:, 0], s=1)
    # plt.scatter(local_ic[7, 1], local_ic[7, 0], s=1)

    fake_pt = [all_traj[7]]
    for i in range(1, 6):
        fake_pt.append(all_traj[7] + [i, i] + np.random.rand(2) * 0.3)
        fake_pt.append(all_traj[7] + [-i, -i] + np.random.rand(2) * 0.3)
        fake_pt.append(all_traj[7] + [i, -i] + np.random.rand(2) * 0.3)
        fake_pt.append(all_traj[7] + [-i, i] + np.random.rand(2) * 0.3)
    fake_pt = np.array(fake_pt)
    fake_pixel = fake_pt[:,[1, 0]] * zoom

    fake_pixel = np.round(fake_pixel + dist).astype(int)
    # plt.scatter(fake_pixel[:, 1], fake_pixel[:, 0], s=1, c='w')

    h, _ = cv2.findHomography(np.array([fake_pixel]), np.array(fake_pt))

    # back_wc = np.matmul(np.concatenate([local_ic, np.ones((len(local_ic), 1))], axis=1), np.transpose(h))
    # back_wc /= np.expand_dims(back_wc[:, 2], 1)
    # back_wc = back_wc[:, :2]
    # (back_wc - all_traj).max()
    local_map = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor()
    ])(Image.fromarray(1 - local_map / 255))

    # return np.expand_dims(1 - local_map / 255, 0), torch.tensor(all_pixel_local), torch.tensor(h).float()
    return local_map, local_ic, torch.tensor(h).float()



def get_local_map_ic(map, all_traj, inv_h_t, zoom=10, radius=8):
    radius = radius * zoom
    context_size = radius * 2

    # from skimage.transform import resize
    # global_map = resize(map, (160, 160))
    # zoom= [160 / map.shape[0], 160 / map.shape[1] ]

    global_map = np.kron(map, np.ones((zoom, zoom)))
    expanded_obs_img = np.full((global_map.shape[0] + context_size, global_map.shape[1] + context_size),
                               False, dtype=np.float32)
    expanded_obs_img[radius:-radius, radius:-radius] = global_map.astype(np.float32)  # 99~-99

    all_pixel = np.matmul(np.concatenate([all_traj, np.ones((len(all_traj), 1))], axis=1), inv_h_t)
    all_pixel /= np.expand_dims(all_pixel[:, 2], 1)
    all_pixel = radius + np.round(all_pixel[:, :2]).astype(int)

    # plt.imshow(expanded_obs_img)
    # plt.scatter(all_pixel[:8, 1], all_pixel[:8, 0], s=1, c='b')
    # plt.scatter(all_pixel[8:, 1], all_pixel[8:, 0], s=1, c='r')
    # plt.show()


    local_map = expanded_obs_img[all_pixel[7, 0] - radius: all_pixel[7, 0] + radius,
                all_pixel[7, 1] - radius: all_pixel[7, 1] + radius]

    '''
    for i in range(len(all_traj)):
        expanded_obs_img[radius + all_pixel[i, 0], radius + all_pixel[i, 1]] = 255

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
        fake_pt.append(all_traj[7] + [i, i] + np.random.rand(2) * 0.3)
        fake_pt.append(all_traj[7] + [-i, -i] + np.random.rand(2) * 0.3)
        fake_pt.append(all_traj[7] + [i, -i] + np.random.rand(2) * 0.3)
        fake_pt.append(all_traj[7] + [-i, i] + np.random.rand(2) * 0.3)
    fake_pt = np.array(fake_pt)

    fake_pixel = np.matmul(np.concatenate([fake_pt, np.ones((len(fake_pt), 1))], axis=1), inv_h_t)
    fake_pixel /= np.expand_dims(fake_pixel[:, 2], 1)
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

    all_pixel_local = np.matmul(np.concatenate([all_traj, np.ones((len(all_traj), 1))], axis=1),
                                np.linalg.pinv(np.transpose(h)))
    all_pixel_local /= np.expand_dims(all_pixel_local[:, 2], 1)
    all_pixel_local = np.round(all_pixel_local).astype(int)[:, :2]

    ##  back to wc validate
    # back_wc = np.matmul(np.concatenate([all_pixel_local, np.ones((len(all_pixel_local), 1))], axis=1), np.transpose(h))
    # back_wc /= np.expand_dims(back_wc[:, 2], 1)
    # back_wc = back_wc[:,:2]
    # (back_wc - all_traj).max()

    #
    # plt.imshow(local_map)
    # plt.scatter(all_pixel_local[:8, 1], all_pixel_local[:8, 0], s=1, c='b')
    # plt.scatter(all_pixel_local[8:, 1], all_pixel_local[8:, 0], s=1, c='r')
    # plt.show()
    # per_step_pixel = np.sqrt(((all_pixel_local[1:] - all_pixel_local[:-1]) ** 2).sum(1)).mean()
    # per_step_wc = np.sqrt(((all_traj[1:] - all_traj[:-1]) ** 2).sum(1)).mean()


    local_map = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor()
    ])(Image.fromarray(1 - local_map / 255))

    # return np.expand_dims(1 - local_map / 255, 0), torch.tensor(all_pixel_local), torch.tensor(h).float()
    return local_map, all_pixel_local, torch.tensor(h).float()


