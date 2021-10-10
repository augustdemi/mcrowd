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

    obs_frames = np.concatenate(obs_frames)
    fut_frames = np.concatenate(fut_frames)
    # map_path = np.concatenate(map_path, 0)
    inv_h_t = np.concatenate(inv_h_t, 0)
    # local_map = np.array(np.concatenate(local_map, 0))
    # local_map = np.concatenate(local_map, 0)
    local_ic = np.concatenate(local_ic, 0)
    local_homo = torch.tensor(np.concatenate(local_homo, 0)).float().to(obs_traj.device)

    local_maps = []
    for seq_local_map in local_map:
        for elt in seq_local_map:
            local_maps.append(np.expand_dims(elt,0))

    # all_traj_st = torch.cat([obs_traj, fut_traj], dim=0)
    # # pos is stdized by mean = last obs step
    # std = all_traj_st[:, :, :2].std(0)
    # all_traj_st[:, :, :2] = (all_traj_st[:,:,:2] - all_traj_st[7, :, :2]) / std
    # all_traj_st[:, :, 2:4] /= std**2
    # all_traj_st[:, :, 4:6] /= std**3
    #
    #
    # out = [
    #     obs_traj, fut_traj, all_traj_st[:8], all_traj_st[8:], seq_start_end,
    #     obs_frames, fut_frames, map_path, inv_h_t,
    #     local_maps, local_ic, local_homo
    # ]

    obs_traj_st = obs_traj.clone()
    # pos is stdized by mean = last obs step
    std = 100
    obs_traj_st[:, :, :2] = (obs_traj_st[:,:,:2] - obs_traj_st[-1, :, :2]) / std
    obs_traj_st[:, :, 2:] /= std
    # print(obs_traj_st.max(), obs_traj_st.min())

    out = [
        obs_traj, fut_traj, obs_traj_st, fut_traj[:,:,2:4] / std, seq_start_end,
        obs_frames, fut_frames, map_path, inv_h_t,
        local_maps, local_ic, local_homo
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
        local_map_size=[]

        self.stats={}
        self.maps={}
        for file in os.listdir(self.map_dir):
            m = imageio.imread(os.path.join(self.map_dir, file)).astype(float)
            m[np.argwhere(m == 1)[:, 0], np.argwhere(m == 1)[:, 1]] = 0
            m[np.argwhere(m == 2)[:, 0], np.argwhere(m == 2)[:, 1]] = 0
            m[np.argwhere(m == 3)[:, 0], np.argwhere(m == 3)[:, 1]] = 1
            m[np.argwhere(m == 4)[:, 0], np.argwhere(m == 4)[:, 1]] = 1
            m[np.argwhere(m == 5)[:, 0], np.argwhere(m == 5)[:, 1]] = 1
            self.maps.update({file.split('.')[0]:m})


        with open(self.data_path, 'rb') as f:
            data = pickle5.load(f)

        data = pd.DataFrame(data)
        scenes = data['sceneId'].unique()
        for s in scenes:
            # if (data_split=='train') and ('hyang_7' not in s):
            if ('nexus_2' in s) or ('hyang_4' in s):
                continue
            # if ('hyang' not in s):
            #     continue
            print(s)
            scene_data = data[data['sceneId'] == s]
            scene_data = scene_data.sort_values(by=['frame', 'trackId'], inplace=False)

            # print(scene_data.shape[0])
            frames = scene_data['frame'].unique().tolist()
            scene_data = np.array(scene_data)
            map_size = self.maps[s + '_mask'].shape
            scene_data[:,2] = np.clip(scene_data[:,2], a_min=None, a_max=map_size[1]-1)
            scene_data[:,3] =  np.clip(scene_data[:,3], a_min=None, a_max=map_size[0]-1)

            # mean = scene_data[:,2:4].astype(float).mean(0)
            # std = scene_data[:,2:4].astype(float).std(0)
            # scene_data[:, 2:4] = (scene_data[:, 2:4] - mean) / std
            # self.stats.update({s: {'mean': mean, 'std': std}})
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
                # if data_split == 'test' and np.concatenate(this_scene_seq).shape[0] > 10:
                #     break


            this_scene_seq = np.concatenate(this_scene_seq)
            # print(s, len(scene_data), this_scene_seq.shape[0])

            '''
            argmax_idx = (per_step_dist * 20).argmax()
            # argmax_idx = 3
            # plt.scatter(this_scene_seq[argmax_idx, 0, :8], this_scene_seq[argmax_idx, 1, :8], s=1, c='b')
            # plt.scatter(this_scene_seq[argmax_idx, 0, 8:], this_scene_seq[argmax_idx, 1, 8:], s=1, c='r')
            # plt.imshow(self.maps[s + '_mask'])
            for i in range(8):
                plt.scatter(this_scene_seq[argmax_idx, 0, i], this_scene_seq[argmax_idx, 1, i], s=4, c='b', alpha=(1-((i+1)/10)))
            for i in range(8,20):
                plt.scatter(this_scene_seq[argmax_idx, 0, i], this_scene_seq[argmax_idx, 1, i], s=4, c='r', alpha=(1-((i)/20)))
            traj = this_scene_seq[argmax_idx].transpose(1, 0)
            np.sqrt(((traj[1:] - traj[:-1]) ** 2).sum(1))
            
            '''
            ### for map
            per_step_dist = []
            for traj in this_scene_seq:
                traj = traj.transpose(1, 0)
                per_step_dist.append(np.sqrt(((traj[1:] - traj[:-1]) ** 2).sum(1)).sum())
            per_step_dist = np.array(per_step_dist)
            # mean_dist = per_step_dist.mean()
            # print(mean_dist)
            per_step_dist = np.clip(per_step_dist, a_min=240, a_max=None)
            # print(per_step_dist.max())
            # print(per_step_dist.mean())
            # local_map_size.extend(np.round(per_step_dist).astype(int) * 13)
            # max_per_step_dist_of_seq = per_step_dist.max()
            # local_map_size.extend([int(max_per_step_dist_of_seq * 13)] * len(this_scene_seq))
            local_map_size.extend(np.round(per_step_dist).astype(int))
            print( self.maps[s + '_mask'].shape, ': ' ,(per_step_dist).max())
            # print(self.maps[s + "_mask"].shape, int(max_per_step_dist_of_seq * 13) * 2)

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
        self.num_seq = len(self.seq_start_end) # = slide (seq. of 16 frames) 수 = 2692
        self.local_map_size = np.stack(local_map_size)
        self.local_ic = [[]] * len(self.obs_frame_num)
        self.local_homo = [[]] * len(self.obs_frame_num)

        print(self.seq_start_end[-1])


    def __len__(self):
        return self.num_seq

    def __getitem__(self, seq_idx):
        start, end = self.seq_start_end[seq_idx]
        global_map = self.maps[self.map_file_name[start] + '_mask']
        inv_h_t = np.expand_dims(np.eye(3), axis=0)

        local_maps =[]
        local_ics =[]
        local_homos =[]
        for index in range(start, end):
            all_traj = torch.cat([self.obs_traj[index, :2, :],  self.pred_traj[index, :2, :]], dim=1).detach().cpu().numpy().transpose((1,0))
            if len(self.local_ic[index]) == 0:
                local_map, local_ic, local_homo = self.get_local_map_ic(global_map, all_traj, zoom=1, radius=self.local_map_size[index], compute_local_homo=True)
                self.local_ic[index] = local_ic
                self.local_homo[index] = local_homo
            else:
                local_map, _, _ = self.get_local_map_ic(global_map, all_traj, zoom=1, radius=self.local_map_size[index])
                local_ic = self.local_ic[index]
                local_homo = self.local_homo[index]

            local_maps.append(local_map)
            local_ics.append(local_ic)
            local_homos.append(local_homo)


        local_ics = np.stack(local_ics)
        local_homos = np.stack(local_homos)

        #########
        out = [
            self.obs_traj[start:end].to(self.device), self.pred_traj[start:end].to(self.device),
            self.obs_frame_num[start:end], self.fut_frame_num[start:end],
            [global_map] * (end-start), [inv_h_t] * (end-start),
            local_maps, local_ics, local_homos
        ]
        return out




    def get_local_map_ic(self, global_map, all_traj, zoom=10, radius=8, compute_local_homo=False):
        radius = radius * zoom
        context_size = radius * 2

        # global_map = np.kron(map, np.ones((zoom, zoom)))
        expanded_obs_img = np.full((global_map.shape[0] + context_size, global_map.shape[1] + context_size),
                                   1, dtype=np.float32)
        expanded_obs_img[radius:-radius, radius:-radius] = global_map.astype(np.float32)  # 99~-99

        # all_pixel = np.matmul(np.concatenate([all_traj, np.ones((len(all_traj), 1))], axis=1), inv_h_t)
        # all_pixel /= np.expand_dims(all_pixel[:, 2], 1)
        # all_pixel = radius + np.round(all_pixel[:, :2]).astype(int)
        all_pixel = all_traj[:,[1,0]]
        all_pixel = radius + np.round(all_pixel).astype(int)

        '''
        plt.imshow(expanded_obs_img)
        plt.scatter(all_pixel[:8, 1], all_pixel[:8, 0], s=1, c='b')
        plt.scatter(all_pixel[8:, 1], all_pixel[8:, 0], s=1, c='r')
        plt.show()
        '''

        local_map = expanded_obs_img[all_pixel[7, 0] - radius: all_pixel[7, 0] + radius,
                    all_pixel[7, 1] - radius: all_pixel[7, 1] + radius]

        all_pixel_local = None
        h = None
        if compute_local_homo:
            fake_pt = [all_traj[7]]
            per_pixel_dist = radius // 10
            # for i in range(10, radius-210, (radius-2)//5):
            for i in range(per_pixel_dist, radius // 2 - per_pixel_dist, per_pixel_dist):
                # print(i)
                fake_pt.append(all_traj[7] + [i, i] + np.random.rand(2) * (per_pixel_dist//2))
                fake_pt.append(all_traj[7] + [-i, -i] + np.random.rand(2) * (per_pixel_dist//2))
                fake_pt.append(all_traj[7] + [i, -i] + np.random.rand(2) * (per_pixel_dist//2))
                fake_pt.append(all_traj[7] + [-i, i] + np.random.rand(2) * (per_pixel_dist//2))
            fake_pt = np.array(fake_pt)

            # fake_pixel = np.matmul(np.concatenate([fake_pt, np.ones((len(fake_pt), 1))], axis=1), inv_h_t)
            # fake_pixel /= np.expand_dims(fake_pixel[:, 2], 1)
            # fake_pixel = radius + np.round(fake_pixel).astype(int)
            fake_pixel = fake_pt[:,[1,0]]
            fake_pixel = radius + np.round(fake_pixel).astype(int)
            '''
            plt.imshow(expanded_obs_img)
            plt.scatter(fake_pixel[:, 1], fake_pixel[:, 0], s=1, c='r')
            '''

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

            '''
            ##  back to wc validate
            back_wc = np.matmul(np.concatenate([all_pixel_local, np.ones((len(all_pixel_local), 1))], axis=1), np.transpose(h))
            back_wc /= np.expand_dims(back_wc[:, 2], 1)
            back_wc = back_wc[:,:2]
            print((back_wc - all_traj).max())
            print(np.sqrt(((back_wc - all_traj)**2).sum(1)).max())
        
        
            plt.imshow(local_map)
            plt.scatter(all_pixel_local[:8, 1], all_pixel_local[:8, 0], s=1, c='b')
            plt.scatter(all_pixel_local[8:, 1], all_pixel_local[8:, 0], s=1, c='r')
            plt.show()
            # per_step_pixel = np.sqrt(((all_pixel_local[1:] - all_pixel_local[:-1]) ** 2).sum(1)).mean()
            # per_step_wc = np.sqrt(((all_traj[1:] - all_traj[:-1]) ** 2).sum(1)).mean()
            '''
            #
            # local_map = resize(local_map, (160, 160))

            # return np.expand_dims(1 - local_map / 255, 0), torch.tensor(all_pixel_local), torch.tensor(h).float()
        return local_map, all_pixel_local, h
