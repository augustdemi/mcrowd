import logging
import os
import math
import pickle
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.append('../')
from utils import derivative_of

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import imageio




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



class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_dir, data_name, data_split, obs_len=8, pred_len=12, skip=1,
            min_ped=0, delim='\t', dt=0.4
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

        self.data_dir = os.path.join(data_dir, data_name, data_split)
        # self.data_dir = '../../datasets/eth/test'
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = 1
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.map_dir = os.path.join(data_dir,'nmap/map/')

        n_state = 6

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []

        obs_frame_num = []
        fut_frame_num = []
        map_file_names = []
        inv_h_ts = []
        local_map_size = []
        deli = '/'
        # deli = '\\'


        for path in all_files:
            print('data path:', path)
            # if 'zara' in path or 'eth' in path or 'hotel' in path:
            # if 'zara' in path or 'hotel' in path or '003' in path:
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

            if map_file_name != '':
                h = np.loadtxt(os.path.join(self.map_dir, map_file_name + '_H.txt'))
                inv_h_t = np.linalg.pinv(np.transpose(h))
                map_file_name = os.path.join(self.map_dir, map_file_name + '_map.png')
            else:
                inv_h_t = np.zeros((3, 3))

            data = read_file(path, self.delim)
            # print('uniq ped: ', len(np.unique(data[:, 1])))


            if 'zara01' in map_file_name:
                frames = (np.unique(data[:, 0]) + 10).tolist()
            else:
                frames = np.unique(data[:, 0]).tolist()

            if data_split == 'test' and data_name !='eth':
                print(len(frames))
                if data_name == 'hotel':
                    idx = 550
                elif 'zara' in data_name:
                    idx = 200
                else:
                    idx= 40
                frames = frames[:idx]

            # print('uniq frames: ', len(frames))
            frame_data = []  # all data per frame
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(
                frames) - self.seq_len + 1) / self.skip))  # seq_len=obs+pred길이씩 잘라서 (input=obs, output=pred)주면서 train시킬것. 그래서 seq_len씩 slide시키면서 총 num_seq만큼의 iteration하게됨

            this_seq_obs = []

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
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[
                                                 0, 0]) - idx  # sliding idx를 빼주는 이유?. sliding이 움직여온 step인 idx를 빼줘야 pad_front=0 이됨. 0보다 큰 pad_front라는 것은 현ped_id가 처음 나타난 frame이 desired first frame보다 더 늦은 경우.
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1  # pad_end까지선택하는 index로 쓰일거라 1더함
                    if pad_end - pad_front != self.seq_len:  # seq_len만큼의 sliding동안 매 프레임마다 agent가 존재하지 않은 데이터였던것.
                        continue
                    ped_ids.append(ped_id)
                    # x,y,x',y',x'',y''
                    x = curr_ped_seq[:, 2]
                    y = curr_ped_seq[:, 3]
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
                    this_seq_obs.append(curr_seq[:num_peds_considered, :2, :8])
                    obs_frame_num.append(np.ones((num_peds_considered, self.obs_len)) * frames[idx:idx + self.obs_len])
                    fut_frame_num.append(
                        np.ones((num_peds_considered, self.pred_len)) * frames[idx + self.obs_len:idx + self.seq_len])
                    # map_file_names.append(num_peds_considered*[map_file_name])
                    map_file_names.append(map_file_name)
                    inv_h_ts.append(inv_h_t)

            ### for map
            if map_file_name == '':
                per_step_dist = []
                for obs_traj in np.concatenate(this_seq_obs):
                    obs_traj = obs_traj.transpose(1, 0)
                    per_step_dist.append(np.sqrt(((obs_traj[1:] - obs_traj[:-1]) ** 2).sum(1)).mean())
                mean_pixel_dist = 0.7
                # argmax = np.concatenate(this_seq_obs)[np.array(per_step_dist).argmax()].transpose(1,0)
                # plt.scatter(argmax[:, 1], argmax[:, 0], s=1)

            else:
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

                two_wc_pts = np.array([[0, 0], [0, 0.7]])
                two_ic_pts = np.matmul(np.concatenate([two_wc_pts, np.ones((len(two_wc_pts), 1))], axis=1),
                                       inv_h_t)
                two_ic_pts /= np.expand_dims(two_ic_pts[:, 2], 1)
                mean_pixel_dist = np.linalg.norm(two_ic_pts[1, :2] - two_ic_pts[0, :2])


                # map_path = os.path.join(self.map_dir, map_file_name + '_map.png')
                # global_map = imageio.imread(map_path)
                # plt.imshow(global_map)
                # argmax = all_pixels[np.array(per_step_dist).argmax()]
                # plt.scatter(argmax[:, 1], argmax[:, 0], s=1)

            per_step_dist = np.array(per_step_dist)
            # max_per_step_dist_of_seq = per_step_dist[np.where(per_step_dist>0.1)[0]].max()
            # max_per_step_dist_of_seq = per_step_dist.max()
            # local_map_size.extend([int(max_per_step_dist_of_seq * 13)] * len(this_seq_obs))
            local_map_size.extend(list((np.clip(per_step_dist, mean_pixel_dist, None) * 16).astype(int)))


        self.num_seq = len(seq_list)  # = slide (seq. of 16 frames) 수 = 2692

        cum_start_idx = [0] + np.cumsum(
            num_peds_in_seq).tolist()  # num_peds_in_seq = 각 slide(16개 frames)별로 고려된 agent수.따라서 len(num_peds_in_seq) = slide 수 = 2692 = self.num_seq
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]  # [(0, 2),  (2, 4),  (4, 7),  (7, 10), ... (32682, 32684),  (32684, 32686)]
        print(self.seq_start_end[-1])

        seq_list = np.concatenate(seq_list, axis=0)  # (32686, 2, 16)
        self.obs_frame_num = np.concatenate(obs_frame_num, axis=0)
        self.fut_frame_num = np.concatenate(fut_frame_num, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = seq_list[:, :, :self.obs_len]
        self.fut_traj = seq_list[:, :, self.obs_len:]
        # frame seq순, 그리고 agent id순으로 쌓아온 데이터에 대한 index를 부여하기 위해 cumsum으로 index생성 ==> 한 슬라이드(16 seq. of frames)에서 고려된 agent의 data를 start, end로 끊어내서 index로 골래내기 위해

        self.map_file_name = map_file_names
        self.local_map_size = local_map_size
        self.inv_h_t = inv_h_ts


        self.local_map = []
        self.local_homo = []
        self.local_ic = []
        self.undated_local_map_size = []

        for seq_i in range(len(self.seq_start_end)):
            start, end = self.seq_start_end[seq_i]
            local_maps = []
            local_ics = []
            local_homos = []
            if self.map_file_name[seq_i] == '':
                zoom = 40
                for idx in range(start, end):
                    all_traj = np.concatenate([self.obs_traj[idx, :2], self.fut_traj[idx, :2]], axis=1).transpose(1, 0)
                    local_map, local_ic, local_h = get_local_map_ic_no_map(all_traj, zoom=zoom,
                                                                           radius=self.local_map_size[idx])
                    local_maps.append(local_map)
                    local_ics.append(local_ic)
                    local_homos.append(local_h)
                local_map_size = np.array(self.local_map_size[start:end]) * 2 * zoom
            else:
                global_map = 255 - imageio.imread(self.map_file_name[seq_i])
                inv_h_t = self.inv_h_t[seq_i]

                for idx in range(start, end):
                    all_traj = np.concatenate([self.obs_traj[idx, :2], self.fut_traj[idx, :2]], axis=1).transpose(1, 0)
                    # plt.imshow(global_map)
                    # plt.scatter(all_traj[:8,0], all_traj[:8,1], s=1, c='b')
                    # plt.scatter(all_traj[8:,0], all_traj[8:,1], s=1, c='r')
                    # plt.show()
                    # eth = 256, zara1 =384 = hotel
                    # students003: 470
                    local_map, local_ic, local_h = get_local_map_ic(global_map, all_traj, inv_h_t, zoom=1,
                                                                    radius=self.local_map_size[idx])
                    local_maps.append(local_map)
                    local_ics.append(local_ic)
                    local_homos.append(local_h)
                local_map_size = np.array(self.local_map_size[start:end]) * 2

                # plt.imshow(local_map[0])
                # plt.scatter(local_ic[:, 1], local_ic[:, 0], s=1)
                # plt.scatter(local_ic[7, 1], local_ic[7, 0], s=1, c='r')
            self.local_map.append(np.stack(local_maps))
            self.local_ic.append(np.stack(local_ics))
            self.local_homo.append(np.stack(local_homos))
            self.undated_local_map_size.append(local_map_size)
        self.local_map = np.concatenate(self.local_map)
        self.local_ic = np.concatenate(self.local_ic)
        self.local_homo = np.concatenate(self.local_homo)
        self.undated_local_map_size = np.concatenate(self.undated_local_map_size)


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
             'local_map_size': self.undated_local_map_size,
             }
        save_path = os.path.join(data_dir, data_name, data_split + '.pickle')
        with open(save_path, 'wb') as handle:
            pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)




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
    fake_pixel = fake_pt[:, [1, 0]] * zoom

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


if __name__ == '__main__':



    traj = TrajectoryDataset(
            data_dir='../../datasets',
            data_name='univ',
            data_split='test',
            skip=0)