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
     local_map, local_ic, local_homo, scale) = zip(*data)
    scale = scale[0]

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
    inv_h_t = np.stack(inv_h_t, 0)
    # local_map = np.array(np.concatenate(local_map, 0))
    # local_map = np.concatenate(local_map, 0)
    local_ic = np.concatenate(local_ic, 0)
    local_homo = torch.tensor(np.concatenate(local_homo, 0)).float().to(obs_traj.device)

    local_maps = []
    for maps in local_map:
        local_maps.extend(maps)


    obs_traj_st = obs_traj.clone()
    # pos is stdized by mean = last obs step
    obs_traj_st[:, :, :2] = (obs_traj_st[:,:,:2] - obs_traj_st[-1, :, :2]) / scale
    obs_traj_st[:, :, 2:] /= scale
    # print(obs_traj_st.max(), obs_traj_st.min())

    out = [
        obs_traj, fut_traj, obs_traj_st, fut_traj[:,:,2:4] / scale, seq_start_end,
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
            self, data_dir, data_split, device='cpu', scale=100
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
        dt=0.4
        min_ped=0
        data_dir = data_dir.replace('\\', '/')


        if data_split == 'train':
            max_num_file = 40
        else:
            max_num_file = 5


        self.seq_len = self.obs_len + self.pred_len


        n_state = 6


        with open(os.path.join(data_dir, data_split + '.txt')) as f:
            files = f.readlines()
            all_files = []
            prev_file = ''
            num_file = 0
            for f in files:
                # if 'biwi' not in f:
                #    continue
                if f.split('\\')[:-1] == prev_file and num_file ==max_num_file:
                    continue
                elif f.split('\\')[:-1] != prev_file:
                    print('/'.join(prev_file))
                    print(num_file)
                    num_file=0
                    prev_file = f.split('\\')[:-1]
                num_file +=1
                all_files.append(f.rstrip().replace('\\', '/'))
            print('/'.join(prev_file))
            print(num_file)




        self.homographys={}
        self.maps={}
        for root, subdirs, files in os.walk(data_dir):
            if len(subdirs)>0:
                continue
            map_file_name = [file_name for file_name in files if 'png' in file_name][0]
            # print(root)
            # print(map_file_name)
            m = imageio.imread(os.path.join(root, map_file_name)).astype(float) / 255
            self.maps.update({root[len(data_dir):].replace('\\', '/')[1:]: m})
            h = read_file(os.path.join(root, map_file_name.replace('png','hom')), ',')
            self.homographys.update({root[len(data_dir):].replace('\\', '/')[1:]: h})

        '''
        fig = plt.figure(figsize=(15, 9))
        i = 0
        for k in self.maps.keys():
            i += 1
            ax = fig.add_subplot(4,5, i)
            ax.imshow(self.maps[k])
            ax.set_title(k, size=7)
            fig.tight_layout()
        fig.tight_layout()
        '''
            # ax.axis('off')


        num_peds_in_seq = []
        seq_list = []

        obs_frame_num = []
        fut_frame_num = []
        data_files =  []

        for path in all_files:
            # print('data path:', path)
            loaded_data = read_file(os.path.join(data_dir, path), delim)

            data = pd.DataFrame(loaded_data)
            data.columns = ['f', 'a', 'pos_x', 'pos_y']
            # data.sort_values(by=['f', 'a'], inplace=True)
            data.sort_values(by=['f', 'a'], inplace=True)

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
            for idx in range(0, num_sequences * skip + 1, skip):
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
                    data_files.append(path)
            cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
            aa = np.array([(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])])
            print(path, aa[-1][1], np.round((aa[:,1]-aa[:,0]).mean(),2))

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

        # self.num_seq = len(self.seq_start_end)
        self.num_seq = len(self.obs_traj)
        self.data_files = np.stack(data_files)
        self.local_ic = [[]] * len(self.obs_traj)
        self.local_homo = [[]] * len(self.obs_traj)
        self.local_map = [[]] * len(self.obs_traj)

        print(self.seq_start_end[-1])

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        # start, end = self.seq_start_end[index]
        start, end = index, index+1
        ss = np.array(self.seq_start_end)
        seq_idx = np.where((index >= ss[:,0]) & (index < ss[:,1]))[0][0]

        key = '/'.join(self.data_files[seq_idx].split('/')[:-1])
        global_map = self.maps[key]
        # global_map = np.expand_dims(global_map, 0).repeat((end-start), axis=0)
        # inv_h_t = np.expand_dims(np.eye(3), 0).repeat((end-start), axis=0)
        inv_h_t = np.linalg.inv(np.transpose(self.homographys[key]))

        seq_all_traj = torch.cat([self.obs_traj[start:end, :2, :],  self.pred_traj[start:end, :2, :]], dim=2).detach().cpu().numpy().transpose((0, 2, 1))

        local_maps = []
        local_ics = []
        local_homos = []

        i = start
        for all_traj in seq_all_traj:
            if len(self.local_ic[i]) == 0:
                map_traj = np.matmul(np.concatenate([all_traj, np.ones((len(all_traj), 1))], 1), inv_h_t)
                map_traj /= np.expand_dims(map_traj[:, 2], 1)
                map_traj = map_traj[:, :2]
                '''
                plt.imshow(global_map)
                plt.scatter(map_traj[:8, 0], map_traj[:8, 1], s=1, c='b')
                plt.scatter(map_traj[8:, 0], map_traj[8:, 1], s=1, c='r')
                '''
                radius = np.sqrt(((map_traj[1:] - map_traj[:-1]) ** 2).sum(1)).mean() * 20
                radius = np.round(radius).astype(int)

                local_map, local_ic, local_homo = self.get_local_map_ic(global_map, inv_h_t, map_traj, all_traj, zoom=30,
                                                                        radius=radius,
                                                                        compute_local_homo=True)
                self.local_map[i] = local_map
                self.local_ic[i] = local_ic
                self.local_homo[i] = local_homo
            else:
                # local_map, _, _ = self.get_local_map_ic(global_map, inv_h_t, map_traj, all_traj, zoom=30, radius=radius)
                local_map = self.local_map[i]
                local_ic = self.local_ic[i]
                local_homo = self.local_homo[i]

            local_maps.append(np.expand_dims(local_map,0))
            local_ics.append(local_ic)
            local_homos.append(local_homo)
            i += 1


        local_ics = np.stack(local_ics)
        local_homos = np.stack(local_homos)

        #########
        out = [
            self.obs_traj[start:end].to(self.device), self.pred_traj[start:end].to(self.device),
            self.obs_frame_num[start], self.fut_frame_num[start],
            self.data_files[seq_idx], inv_h_t,
            local_maps, local_ics, local_homos, self.scale
        ]
        return out


    def get_local_map_ic(self, global_map, inv_h_t, map_traj, all_traj, zoom=10, radius=8, compute_local_homo=False):
            radius = radius * zoom
            radius = max(radius, 128)
            context_size = radius * 2

            global_map = np.kron(global_map, np.ones((zoom, zoom)))
            expanded_obs_img = np.full((global_map.shape[0] + context_size, global_map.shape[1] + context_size),
                                       0, dtype=np.float32)
            expanded_obs_img[radius:-radius, radius:-radius] = global_map.astype(np.float32)  # 99~-99

            all_pixel = map_traj[:,[1,0]] * zoom
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
                for i in [0.5, 1, 1.5]:
                    fake_pt.append(all_traj[7] + [i, i] + np.random.rand(2) * 0.3)
                    fake_pt.append(all_traj[7] + [-i, -i] + np.random.rand(2) * 0.3)
                    fake_pt.append(all_traj[7] + [i, -i] + np.random.rand(2) * 0.3)
                    fake_pt.append(all_traj[7] + [-i, i] + np.random.rand(2) * 0.3)
                fake_pt = np.array(fake_pt)

                fake_pixel = np.matmul(np.concatenate([fake_pt[:,[1,0]], np.ones((len(fake_pt), 1))], axis=1), inv_h_t)
                fake_pixel /= np.expand_dims(fake_pixel[:, 2], 1)
                fake_pixel = fake_pixel[:, :2] * zoom
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

                h, _ = cv2.findHomography(np.array(fake_local_pixel), fake_pt)
                # h, _ = cv2.findHomography(np.array(fake_local_pixel)[:,[1,0]], np.array(fake_pt))

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
            return local_map, all_pixel_local, h
