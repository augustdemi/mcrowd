#https://github.com/Marchetz/KITTI-trajectory-prediction.git
import numpy as np
import torch
import torch.utils.data as data
import cv2
import json
import os
import matplotlib.pyplot as plt

from utils import derivative_of

def seq_collate(data):
    (obs_seq_list, pred_seq_list,
     videos, classes, global_map, homo,
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

    videos = np.stack(videos)
    classes = np.stack(classes)
    # map_path = np.concatenate(map_path, 0)
    # local_map = np.array(np.concatenate(local_map, 0))
    # local_map = np.concatenate(local_map, 0)
    local_ic = np.concatenate(local_ic, 0)
    local_homo = torch.tensor(np.stack(local_homo, 0)).float().to(obs_traj.device)



    obs_traj_st = obs_traj.clone()
    # pos is stdized by mean = last obs step
    obs_traj_st[:, :, :2] = (obs_traj_st[:,:,:2] - obs_traj_st[-1, :, :2]) / scale
    obs_traj_st[:, :, 2:] /= scale
    # print(obs_traj_st.max(), obs_traj_st.min())

    out = [
        obs_traj, fut_traj, obs_traj_st, fut_traj[:,:,2:4] / scale, seq_start_end,
        videos, classes, global_map, homo[0],
        local_map, local_ic, local_homo
    ]

    return tuple(out)

class TrackDataset(data.Dataset):
    """
    Dataset class for KITTI.

    The building class(3) is merged into the background class
    0:background, 1:street, 2:sidewalk, 3: vegetation
    """
    def __init__(self, data_dir, data_split, device):
        json_dataset = os.path.join(data_dir, 'dataset_kitti_' + data_split +'.json')
        tracks = json.load(open(json_dataset))

        self.device = device

        self.index = []
        self.pasts = []             # [len_past, 2]
        self.futures = []           # [len_future, 2]
        # self.positions_in_map = []  # position in complete scene
        self.rotation_angles = []   # trajectory angle in complete scene
        self.scenes = []            # [360, 360, 1]
        self.videos = []            # '0001'
        self.classes = []           # 'Car'
        self.num_vehicles = []      # 0 is ego-vehicle, >0 other agents
        self.step_sequences = []
        self.obs_traj = []
        self.pred_traj = []
        self.homo = np.array([[2, 0, 180], [0, 2, 180], [0, 0, 1]])
        self.local_map_size = []
        self.obs_len = 20

        self.scale=1
        self.zoom = zoom=4
        self.homo = np.array([[  2*zoom,   0, 180*zoom],
                       [  0,   2*zoom, 180*zoom],
                       [  0,   0,   1]])
        dt=0.1

        scene_tracks = {}
        for map_file in os.listdir(data_dir + '/maps'):
            video = map_file.split('drive_')[1].split('_sync')[0]
            scene_track = cv2.imread(os.path.join(data_dir, 'maps', map_file), 0)
            scene_track[np.where(scene_track == 3)] = 0
            scene_track[np.where(scene_track == 0)] = 3
            scene_track[np.where(scene_track == 1)] = 0
            scene_track[np.where(scene_track == 4)] = 1
            scene_tracks.update({video: scene_track})


        # Preload data
        for t in tracks.keys():

            past = np.asarray(tracks[t]['past'])
            future = np.asarray(tracks[t]['future'])
            position_in_map = np.asarray(tracks[t]['position_in_map'])
            rotation_angle = tracks[t]['angle_rotation']
            video = tracks[t]['video']
            class_vehicle = tracks[t]['class']
            num_vehicle = tracks[t]['num_vehicle']
            step_sequence = tracks[t]['step_sequence']

            scene_track = scene_tracks[video]
            scene_track = scene_track[
                                      int(position_in_map[1]) * 2 - 180:int(position_in_map[1]) * 2 + 180,
                                      int(position_in_map[0]) * 2 - 180:int(position_in_map[0]) * 2 + 180]

            matRot_scene = cv2.getRotationMatrix2D((180, 180), rotation_angle, 1)
            scene_track = cv2.warpAffine(scene_track, matRot_scene,
                                         (scene_track.shape[0], scene_track.shape[1]),
                                         borderValue=0,
                                         flags=cv2.INTER_NEAREST)

            self.index.append(t)
            self.pasts.append(past)
            self.futures.append(future)

            curr_ped_seq = np.concatenate([past, future])
            x = curr_ped_seq[:, 0].astype(float)
            y = curr_ped_seq[:, 1].astype(float)
            vx = derivative_of(x, dt)
            vy = derivative_of(y, dt)
            ax = derivative_of(vx, dt)
            ay = derivative_of(vy, dt)
            states =np.stack([x, y, vx, vy, ax, ay])
            self.obs_traj.append(states[:,:self.obs_len])
            self.pred_traj.append(states[:,self.obs_len:])

            # self.positions_in_map.append(position_in_map)
            self.rotation_angles.append(rotation_angle)
            self.videos.append(video)
            self.classes.append(class_vehicle)
            self.num_vehicles.append(num_vehicle)
            self.step_sequences.append(step_sequence)
            self.scenes.append(scene_track)


            traj = curr_ped_seq
            local_map_size = np.sqrt((((traj[1:] - traj[:-1])*2) ** 2).sum(1)).mean() * 60
            local_map_size = np.clip(local_map_size, a_min=32, a_max=None)
            self.local_map_size.append(np.round(local_map_size).astype(int))


        self.obs_traj = torch.from_numpy(np.stack(self.obs_traj)).type(torch.float)
        self.pred_traj = torch.from_numpy(np.stack(self.pred_traj)).type(torch.float)
        self.local_ic = [[]] * len(self.obs_traj)
        self.local_homo = [[]] * len(self.obs_traj)



    def __len__(self):
        return len(self.obs_traj)

    def __getitem__(self, index):
        global_map = self.scenes[index]
        all_traj = torch.cat([self.obs_traj[index, :2, :], self.pred_traj[index, :2, :]],
                             dim=1).detach().cpu().numpy().transpose((1, 0))

        '''
        h = np.array([[2, 0, 180], [0, 2, 180], [0, 0, 1]])
        all_pixel_local = np.matmul(np.concatenate([all_traj, np.ones((len(all_traj), 1))], axis=1),
                                    np.transpose(h))
        all_pixel_local /= np.expand_dims(all_pixel_local[:, 2], 1)
        # all_pixel_local = np.round(all_pixel_local).astype(int)[:, :2]
        plt.imshow(global_map)
        plt.scatter(all_pixel_local[:self.obs_len, 0], all_pixel_local[:self.obs_len, 1], s=1, c='blue')
        plt.scatter(all_pixel_local[self.obs_len:, 0], all_pixel_local[self.obs_len:, 1], s=1, c='red')
        '''

        if len(self.local_ic[index]) == 0:
            local_map, local_ic, local_homo = self.get_local_map_ic(global_map, all_traj, zoom=self.zoom,
                                                                    radius=self.local_map_size[index],
                                                                    compute_local_homo=True)
            self.local_ic[index] = local_ic
            self.local_homo[index] = local_homo
        else:
            local_map, _, _ = self.get_local_map_ic(global_map, all_traj, zoom=self.zoom, radius=self.local_map_size[index])
            local_ic = self.local_ic[index]
            local_homo = self.local_homo[index]


        '''
        ##  back to wc validate
        back_wc = np.matmul(np.concatenate([all_pixel_local, np.ones((len(all_pixel_local), 1))], axis=1), np.transpose(h))
        back_wc /= np.expand_dims(back_wc[:, 2], 1)
        back_wc = back_wc[:,:2]
        print((back_wc - all_traj).max())
        print(np.sqrt(((back_wc - all_traj)**2).sum(1)).max())
        '''

        #########
        out = [
            self.obs_traj[index].to(self.device).unsqueeze(0), self.pred_traj[index].to(self.device).unsqueeze(0),
            self.videos[index], self.classes[index],
            self.scenes[index], self.homo,
            local_map, np.expand_dims(local_ic, axis=0), local_homo, self.scale
        ]
        return out

    def get_local_map_ic(self, global_map, all_traj, zoom=8, radius=8, compute_local_homo=False):
        radius = radius * zoom
        context_size = radius * 2

        global_map = np.kron(global_map, np.ones((zoom, zoom)))
        expanded_obs_img = np.full((global_map.shape[0] + context_size, global_map.shape[1] + context_size),
                                   0, dtype=np.float32)
        expanded_obs_img[radius:-radius, radius:-radius] = global_map.astype(np.float32)  # 99~-99

        all_pixel = np.matmul(np.concatenate([all_traj, np.ones((len(all_traj), 1))], axis=1),
                                    np.transpose(self.homo))
        all_pixel /= np.expand_dims(all_pixel[:, 2], 1)
        all_pixel = all_pixel[:,[1,0]]
        all_pixel = radius + np.round(all_pixel).astype(int)
        '''
        plt.imshow(expanded_obs_img)
        plt.scatter(all_pixel[:self.obs_len, 1], all_pixel[:self.obs_len, 0], s=1, c='b')
        plt.scatter(all_pixel[self.obs_len:, 1], all_pixel[self.obs_len:, 0], s=1, c='r')
        plt.show()
        '''

        local_map = expanded_obs_img[all_pixel[self.obs_len-1, 0] - radius: all_pixel[self.obs_len-1, 0] + radius,
                    all_pixel[self.obs_len-1, 1] - radius: all_pixel[self.obs_len-1, 1] + radius]

        all_pixel_local = None
        h = None
        if compute_local_homo:
            fake_pt = [all_traj[self.obs_len-1]]
            # for i in range(1, 6):
            #     fake_pt.append(all_traj[self.obs_len-1] + [i, i] + np.random.rand(2) * 0.3)
            #     fake_pt.append(all_traj[self.obs_len-1] + [-i, -i] + np.random.rand(2) * 0.3)
            #     fake_pt.append(all_traj[self.obs_len-1] + [i, -i] + np.random.rand(2) * 0.3)
            #     fake_pt.append(all_traj[self.obs_len-1] + [-i, i] + np.random.rand(2) * 0.3)
            per_pixel_dist = (radius//zoom) // 10
            for i in range(per_pixel_dist, (radius//zoom) // 2 - per_pixel_dist, per_pixel_dist):
                fake_pt.append(all_traj[self.obs_len-1] + [i, i] + np.random.rand(2) * (per_pixel_dist // 2))
                fake_pt.append(all_traj[self.obs_len-1] + [-i, -i] + np.random.rand(2) * (per_pixel_dist // 2))
                fake_pt.append(all_traj[self.obs_len-1] + [i, -i] + np.random.rand(2) * (per_pixel_dist // 2))
                fake_pt.append(all_traj[self.obs_len-1] + [-i, i] + np.random.rand(2) * (per_pixel_dist // 2))
            fake_pt = np.array(fake_pt)

            fake_pixel = np.matmul(np.concatenate([fake_pt, np.ones((len(fake_pt), 1))], axis=1), np.transpose(self.homo))
            fake_pixel /= np.expand_dims(fake_pixel[:, 2], 1)
            fake_pixel = fake_pixel[:,[1,0]]
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
            plt.scatter(all_pixel_local[:self.obs_len, 1], all_pixel_local[:self.obs_len, 0], s=1, c='b')
            plt.scatter(all_pixel_local[self.obs_len:, 1], all_pixel_local[self.obs_len:, 0], s=1, c='r')
            plt.show()
            # per_step_pixel = np.sqrt(((all_pixel_local[1:] - all_pixel_local[:-1]) ** 2).sum(1)).mean()
            # per_step_wc = np.sqrt(((all_traj[1:] - all_traj[:-1]) ** 2).sum(1)).mean()
            '''
            #
            # local_map = resize(local_map, (160, 160))

            # return np.expand_dims(1 - local_map / 255, 0), torch.tensor(all_pixel_local), torch.tensor(h).float()
        return local_map, all_pixel_local, h
