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
     obs_frames, fut_frames, past_obst, fut_obst, map_path, inv_h_t) = zip(*data)

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

    map_path = np.array(map_path)
    inv_h_t = np.array(inv_h_t)

    mean = torch.zeros_like(obs_traj[0]).type(torch.FloatTensor).to(obs_traj.device)
    mean[:,:2] = obs_traj[-1,:,:2]
    std = torch.tensor([3, 3, 2, 2, 1, 1]).type(torch.FloatTensor).to(obs_traj.device)

    out = [
        obs_traj, pred_traj, (obs_traj - mean) / std, pred_traj_rel / 2, seq_start_end,
        obs_frames, fut_frames, past_obst, fut_obst, map_path, inv_h_t
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

def crop(map, target_pos, inv_h_t, context_size=198):
    expanded_obs_img = np.full((map.shape[0] + context_size, map.shape[1] + context_size), False, dtype=np.float32)
    expanded_obs_img[context_size//2:-context_size//2, context_size//2:-context_size//2] = map.astype(np.float32) # 99~-99

    #### TEST
    # zara1 data points
    # pts = [[4.1068, 7.5457], [3.90560103932, 7.51993162917], [3.70418592941, 7.493917711], [3.52529058623, 7.48795121601], [3.34639524305, 7.48198472102],  [3.16749989986, 7.47601822602],
    #        [2.98860455668,	7.47005173103], [2.95282548805,	7.41921719369], [2.91704641941, 7.36838265635], [2.98860455668, 7.47005173103], [2.95282548805, 7.41921719369]]
    # pts = [[3.71976034752, 3.09875883948], [3.26957547803, 3.12095420085], [2.81918014343, 3.14314956223], [2.36899527394,3.1653449236], [2.03098830789,	3.18754028497], [1.76790692085,	3.20997430615],
    #        [1.50482553382,	3.23240832732], [1.34087321342,	3.2483985339], [1.32571972553, 3.25484234849]]
    # pts = [[11.5907345173, 5.95718726065], [11.1112949976, 5.86434859856], [10.6356438498, 5.81518467982], [10.1688322367, 5.86721251616], [9.70181015841, 5.91947901229], [9.23436714993, 5.97890530242],
    #        [8.76250437415, 6.10229241887], [8.29085206348, 6.22567953533], [7.82109393879, 6.32734861]]
    # import cv2
    # cap = cv2.VideoCapture('D:\crowd/ucy_original\data/crowds_zara01.avi')
    # cap.set(1, 4390)
    # _, frame = cap.read()
    # plt.imshow(frame)
    #
    # target_pixel = np.matmul(np.concatenate([pts, np.ones((len(pts), 1))], axis=1), inv_h_t)
    # target_pixel /= np.expand_dims(target_pixel[:, 2], 1)
    # target_pixel = target_pixel[:,:2]
    # img_pts = context_size//2 + np.round(target_pixel).astype(int)
    #
    # plt.imshow(expanded_obs_img)
    # for p in range(len(img_pts)):
    #     plt.scatter(img_pts[p][1], img_pts[p][0], c='r', s=1)
    #     expanded_obs_img[img_pts[p][0], img_pts[p][1]] = 255
    # plt.show()
    #########

    target_pos = np.expand_dims(target_pos, 0)
    target_pixel = np.matmul(np.concatenate([target_pos, np.ones((len(target_pos), 1))], axis=1), inv_h_t)
    target_pixel /= np.expand_dims(target_pixel[:, 2], 1)
    target_pixel = target_pixel[:,:2]
    # plt.imshow(map)
    # plt.scatter(target_pixel[0][1], target_pixel[0][0], c='r', s=1)
    #
    #
    # ss = target_pos.squeeze(0)
    # t1 = torch.tensor([6,0]) + ss
    # t2 = torch.tensor([0,6]) + ss
    # t3 = torch.tensor([0,-6]) + ss
    # t4 = torch.tensor([-6, 0]) + ss
    # tt = torch.stack([t1, t2, t3, t4])
    # tt_pixel = np.matmul(np.concatenate([tt, np.ones((len(tt), 1))], axis=1), inv_h_t)
    # tt_pixel /= np.expand_dims(tt_pixel[:, 2], 1)
    # tt_pixel = tt_pixel[:,:2]
    #
    # plt.imshow(map)
    # plt.scatter(target_pixel[0][1], target_pixel[0][0], c='r', s=1)
    # for i in range(4):
    #     plt.scatter(tt_pixel[i][1], tt_pixel[i][0], c='b', s=1)
    # plt.show()

    img_pts = context_size//2 + np.round(target_pixel).astype(int)
    # plt.imshow(expanded_obs_img)
    # plt.scatter(img_pts[0][1], img_pts[0][0], c='r', s=1)

    nearby_area = context_size//2 - 10
    if img_pts[0][0] < nearby_area:
        img_pts[0][0] = nearby_area
        print(target_pos[0])
    elif img_pts[0][0] > expanded_obs_img.shape[0] - nearby_area:
        img_pts[0][0] = expanded_obs_img.shape[0] - nearby_area
        print(target_pos[0])

    if img_pts[0][1] < nearby_area :
        img_pts[0][1] = nearby_area
        print(target_pos[0])
    elif img_pts[0][1] > expanded_obs_img.shape[1] - nearby_area:
        img_pts[0][1] = expanded_obs_img.shape[1] - nearby_area
        print(target_pos[0])

    cropped_img = np.stack([expanded_obs_img[img_pts[i, 0] - nearby_area : img_pts[i, 0] + nearby_area,
                                      img_pts[i, 1] - nearby_area : img_pts[i, 1] + nearby_area]
                      for i in range(target_pos.shape[0])], axis=0)


    cropped_img[0, nearby_area, nearby_area] = 255
    # plt.imshow(cropped_img[0])
    return cropped_img



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
        self.map_dir =  '../datasets/nmap/map/'

        n_pred_state=2
        n_state=6

        self.context_size=context_size
        self.resize=resize

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []

        seq_past_obst_list = []
        seq_fut_obst_list = []
        obs_frame_num = []
        fut_frame_num = []
        map_file_names=[]
        deli = '\\'

        for path in all_files:
            print('data path:', path)
            # if 'zara' in path or 'eth' in path or 'hotel' in path:
            if 'zara' in path or 'hotel' in path:
                continue
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
                map_file_name = ''

            print('map path: ', map_file_name)


            data = read_file(path, delim)
            # print('uniq ped: ', len(np.unique(data[:, 1])))


            if 'zara01' in map_file_name:
                frames = (np.unique(data[:, 0]) + 10).tolist()
            else:
                frames = np.unique(data[:, 0]).tolist()

            df = []
            # print('uniq frames: ', len(frames))
            frame_data = [] # all data per frame
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip)) # seq_len=obs+pred????????? ????????? (input=obs, output=pred)????????? train?????????. ????????? seq_len??? slide???????????? ??? num_seq????????? iteration?????????

            # all frames??? seq_len(kernel size)????????? sliding????????? ??????. ?????? skip = stride.
            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0) # frame??? seq_len????????? ????????? ?????? = curr_seq_data. ??? frame??? ?????? ?????????(agent)?????? ????????? ??????. ????????? ??? ???????????? ????????? 4(frame #, agent id, pos_x, pos_y)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1]) # unique agent id

                curr_seq_rel = np.zeros((len(peds_in_curr_seq), n_pred_state, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), n_state, self.seq_len))
                num_peds_considered = 0
                ped_ids = []
                for _, ped_id in enumerate(peds_in_curr_seq): # current frame sliding??? ????????? ??? agent??? ??????
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :] # frame#, agent id, pos_x, pos_y
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx # sliding idx??? ????????? ???????. sliding??? ???????????? step??? idx??? ????????? pad_front=0 ??????. 0?????? ??? pad_front?????? ?????? ???ped_id??? ?????? ????????? frame??? desired first frame?????? ??? ?????? ??????.
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1 # pad_end?????????????????? index??? ???????????? 1??????
                    if pad_end - pad_front != self.seq_len: # seq_len????????? sliding?????? ??? ??????????????? agent??? ???????????? ?????? ??????????????????.
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

                    ### others
                    per_frame_past_obst = []
                    per_frame_fut_obst = []
                    if map_file_name is '':
                        per_frame_past_obst = [[]] * self.obs_len
                        per_frame_fut_obst = [[]] * self.pred_len
                    else:
                        curr_obst_seq = curr_seq_data[curr_seq_data[:, 1] != ped_id, :] # frame#, agent id, pos_x, pos_y
                        i=0
                        for frame in np.unique(curr_ped_seq[:,0]): # curr_ped_seq??? continue??? ?????????????????? ????????? 20???
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



                if num_peds_considered > min_ped: # ????????? ????????? sliding(16???)?????? ????????? agent?????? min_ped?????? ?????? ???????????? ???????????? ??? slide???????????? ??????
                    num_peds_in_seq.append(num_peds_considered)
                    # ?????? list??? initialize??? peds_in_curr_seq?????? ???????????????, ????????? ???????????? slide??? agent??? ????????? append ????????? ????????? num_peds_considered????????? ????????? ???
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
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


        self.num_seq = len(seq_list) # = slide (seq. of 16 frames) ??? = 2692
        seq_list = np.concatenate(seq_list, axis=0) # (32686, 2, 16)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        self.obs_frame_num = np.concatenate(obs_frame_num, axis=0)
        self.fut_frame_num = np.concatenate(fut_frame_num, axis=0)
        self.past_obst = seq_past_obst_list
        self.fut_obst = seq_fut_obst_list

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        # frame seq???, ????????? agent id????????? ????????? ???????????? ?????? index??? ???????????? ?????? cumsum?????? index?????? ==> ??? ????????????(16 seq. of frames)?????? ????????? agent??? data??? start, end??? ???????????? index??? ???????????? ??????
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist() # num_peds_in_seq = ??? slide(16??? frames)?????? ????????? agent???.????????? len(num_peds_in_seq) = slide ??? = 2692 = self.num_seq
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ] # [(0, 2),  (2, 4),  (4, 7),  (7, 10), ... (32682, 32684),  (32684, 32686)]
        self.map_file_name = map_file_names



    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        map_file_name = self.map_file_name[index]
        if map_file_name is not '':
            map_path = os.path.join(self.map_dir, map_file_name + '_map.png')
            map = imageio.imread(map_path)
            h = np.loadtxt(os.path.join(self.map_dir, map_file_name + '_H.txt'))
            inv_h_t = np.linalg.pinv(np.transpose(h))

            # target_pos =  torch.cat([self.obs_traj[start:end][i, :2], self.pred_traj[start:end][i, :2]], dim=1)
            # target_pos = torch.transpose(target_pos, 1,0)
            # target_pixel = np.matmul(np.concatenate([target_pos, np.ones((len(target_pos), 1))], axis=1), inv_h_t)
            # target_pixel /= np.expand_dims(target_pixel[:, 2], 1)
            # target_pixel = target_pixel[:, :2]
            # plt.imshow(map)
            # for i in range(20):
            #     if i == 7:
            #         plt.scatter(target_pixel[i][1], target_pixel[i][0], c='r', s=3)
            #     else:
            #         plt.scatter(target_pixel[i][1], target_pixel[i][0], c='b', s=1)
            # plt.show()


            past_map_obst = []
            past_obst = self.past_obst[start:end]
            for i in range(len(past_obst)):  # len(past_obst) = batch
                seq_map = []
                for t in range(self.obs_len):
                    cp_map = map.copy()
                    # gt_real = past_obst[i][t]
                    # # mark the obstacle pedestrians
                    # if len(gt_real) > 0:
                    #     gt_pixel = np.matmul(np.concatenate([gt_real, np.ones((len(gt_real), 1))], axis=1), inv_h_t)
                    #     gt_pixel /= np.expand_dims(gt_pixel[:, 2], 1)
                    #     # mark all pixel size
                    #     for p in np.round(gt_pixel)[:, :2].astype(int):
                    #         x = range(max(p[0] - pixel_distance, 0), min(p[0] + pixel_distance + 1, map.shape[0]))
                    #         y = range(max(p[1] - pixel_distance, 0), min(p[1] + pixel_distance + 1, map.shape[1]))
                    #         idx = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
                    #         within_dist_idx = idx[np.linalg.norm(np.ones_like(idx)*p - idx, ord=2, axis=1) < pixel_distance]
                    #         cp_map[within_dist_idx[:,0], within_dist_idx[:,1]] = 255
                    # crop the map near the target pedestrian
                    cp_map = crop(cp_map, self.obs_traj[start:end][i,:2,t], inv_h_t, self.context_size)
                    cp_map = transform(cp_map, self.resize) / 255.0
                    seq_map.append(cp_map)
                past_map_obst.append(np.stack(seq_map))

            past_map_obst = np.stack(past_map_obst) # (batch(start-end), 8, 1, map_size,map_size)
            past_map_obst = torch.from_numpy(past_map_obst)

            fut_map_obst = []
            fut_obst = self.fut_obst[start:end]
            for i in range(len(fut_obst)):
                seq_map = []
                for t in range(self.pred_len):
                    cp_map = map.copy()
                    # gt_real = fut_obst[i][t]
                    # if len(gt_real) > 0:
                    #     gt_pixel = np.matmul(np.concatenate([gt_real, np.ones((len(gt_real), 1))], axis=1), inv_h_t)
                    #     gt_pixel /= np.expand_dims(gt_pixel[:, 2], 1)
                    #     for p in np.round(gt_pixel)[:, :2].astype(int):
                    #         x = range(max(p[0] - pixel_distance, 0), min(p[0] + pixel_distance + 1, map.shape[0]))
                    #         y = range(max(p[1] - pixel_distance, 0), min(p[1] + pixel_distance + 1, map.shape[1]))
                    #         idx = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
                    #         within_dist_idx = idx[
                    #             np.linalg.norm(np.ones_like(idx) * p - idx, ord=2, axis=1) < pixel_distance]
                    #         cp_map[within_dist_idx[:, 0], within_dist_idx[:, 1]] = 255
                    #         # crop the map near the target pedestrian
                    cp_map = crop(cp_map, self.pred_traj[start:end][i, :2, t], inv_h_t, self.context_size)
                    cp_map = transform(cp_map, self.resize) / 255.0
                    seq_map.append(cp_map)
                fut_map_obst.append(np.stack(seq_map))
            fut_map_obst = np.stack(fut_map_obst)  # (batch(start-end), 8, 1, 128,128)
            fut_map_obst = torch.from_numpy(fut_map_obst)
        else: # map is not available
            map_path = inv_h_t = None

            past_map_obst = torch.zeros(end - start, self.obs_len, 1, self.resize, self.resize)
            past_map_obst[:, :, 0, 31,31] =0.0144
            past_map_obst[:, :, 0, 31,32] =0.0336
            past_map_obst[:, :, 0, 32,31] =0.0336
            past_map_obst[:, :, 0, 32,32] =0.0784
            fut_map_obst = torch.zeros(end - start, self.pred_len, 1, self.resize, self.resize)
            fut_map_obst[:, :, 0, 31, 31] = 0.0144
            fut_map_obst[:, :, 0, 31, 32] = 0.0336
            fut_map_obst[:, :, 0, 32, 31] = 0.0336
            fut_map_obst[:, :, 0, 32, 32] = 0.0784


        # image = transforms.Compose([
        #     transforms.ToTensor()
        # ])(image)


        ## real frame img
        # import cv2
        # fig, ax = plt.subplots()
        # cap = cv2.VideoCapture(
        #     'D:\crowd\ewap_dataset\seq_eth\seq_eth.avi')
        # cap.set(1, self.obs_frame_num[start:end][i][t])
        # _, frame = cap.read()
        # ax.imshow(frame)
        #
        # # plt.imshow(cp_map)
        # # fake=np.array([[-2.37,  6.54]])
        # fake = np.expand_dims(self.obs_traj[start:end][i,:,t],0)
        # fake = np.concatenate([fake, np.ones((len(fake), 1))], axis=1)
        # fake_pixel = np.matmul(fake, self.inv_h_t)
        # fake_pixel /= np.expand_dims(fake_pixel[:, 2], 1)
        # plt.scatter(fake_pixel[0,1], fake_pixel[0,0], c='r', s=1)
        # np.linalg.norm(gt_pixel-fake_pixel,2) #  2.5415


        out = [
            self.obs_traj[start:end, :].to(self.device) , self.pred_traj[start:end, :].to(self.device),
            self.obs_traj_rel[start:end, :].to(self.device), self.pred_traj_rel[start:end, :].to(self.device),
            self.obs_frame_num[start:end], self.fut_frame_num[start:end],
            past_map_obst.to(self.device), fut_map_obst.to(self.device), map_path, inv_h_t
        ]
        return out