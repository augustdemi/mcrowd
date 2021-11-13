import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from scipy.stats import gaussian_kde
import argparse
import os
import pickle
import matplotlib.pyplot as plt

def compute_ade(predicted_trajs, gt_traj):
    '''
    :param predicted_trajs: (# sampling, # sequence, # step, # coordinate)
    :param gt_traj: (# sampling, # sequence, # step, # coordinate)
    :return: minADE, avgADE, stdADE
    '''
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    ade = np.mean(error, axis=-1)
    min_ade = ade.min(axis=0)
    avg_ade = ade.mean(axis=0)
    std_ade = ade.std(axis=0)
    return min_ade.mean(), avg_ade.mean(), std_ade.mean()


def compute_fde(predicted_trajs, gt_traj):
    '''
    :param predicted_trajs: (# sampling, # sequence, # coordinate)
    :param gt_traj: (# sampling, # sequence, # coordinate)
    :return: minADE, avgADE, stdADE
    '''
    # final_error = np.linalg.norm(predicted_trajs[:, :, -1] - gt_traj[-1], axis=-1)
    final_error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    min_fde = final_error.min(axis=0)
    avg_fde = final_error.mean(axis=0)
    std_fde = final_error.std(axis=0)
    return min_fde.mean(), avg_fde.mean(), std_fde.mean()


def compute_ECFL(output_traj, binary_navmaps, local_homo):
    '''
    :param output_traj: (# scenes, # samples, # frames, # coordinates) # all sample number, 20, 12, 2
    :param binary_navmaps: (# scenes, # height/y, # width/x)
        1 indicates navigable; 0 indicates non-navigable
    :return: avgECFL
    '''

    ecfl = 0.0
    for i in range(output_traj.shape[0]):
        for k in range(output_traj.shape[1]):
            collided = False
            wc = output_traj[i,k]
            all_pixel_local = np.matmul(np.concatenate([wc, np.ones((len(wc), 1))], axis=1),
                                        np.linalg.pinv(np.transpose(local_homo[i])))
            all_pixel_local /= np.expand_dims(all_pixel_local[:, 2], 1)
            all_pixel_local = np.round(all_pixel_local).astype(int)[:, :2]

            for t in range(output_traj.shape[2]):
                pos = all_pixel_local[t]
                if binary_navmaps[i, pos[0], pos[1]] == 0:
                    collided = True
                    break

            if not collided:
                ecfl += 1.0 / output_traj.shape[1]

    return ecfl / output_traj.shape[0]


import imageio

def compute_ECFL_map(output_traj, map_path):
    '''
    :param output_traj: (# scenes, # samples, # frames, # coordinates) # all sample number, 20, 12, 2
    :param binary_navmaps: (# scenes, # height/y, # width/x)
        1 indicates navigable; 0 indicates non-navigable
    :return: avgECFL
    '''
    map_dir = 'D:\crowd\datasets\SDD\SDD_semantic_maps/test_masks'
    maps = {}
    for file in os.listdir(map_dir):
        m = imageio.imread(os.path.join(map_dir, file)).astype(float)
        m[np.argwhere(m == 1)[:, 0], np.argwhere(m == 1)[:, 1]] = 1
        m[np.argwhere(m == 2)[:, 0], np.argwhere(m == 2)[:, 1]] = 1
        m[np.argwhere(m == 3)[:, 0], np.argwhere(m == 3)[:, 1]] = 0
        m[np.argwhere(m == 4)[:, 0], np.argwhere(m == 4)[:, 1]] = 1
        m[np.argwhere(m == 5)[:, 0], np.argwhere(m == 5)[:, 1]] = 1
        maps.update({file.split('.')[0]: m})

    ecfl = 0.0
    for i in range(output_traj.shape[0]):
        binary_navmaps = maps[map_path[i]]
        for k in range(output_traj.shape[1]):
            collided = False
            wc = output_traj[i,k]

            # plt.imshow(m)
            # plt.scatter(wc[:, 0], wc[:, 1], c='r', s=2)
            for t in range(output_traj.shape[2]):
                pos = wc[t].astype(int)
                if pos[1] < 0 or pos[1] >= binary_navmaps.shape[0] or pos[0] < 0 or pos[0] >= binary_navmaps.shape[1]:
                    collided = True
                    break
                if binary_navmaps[pos[1], pos[0]] == 0:
                    collided = True
                    break

            if not collided:
                ecfl += 1.0 / output_traj.shape[1]

    return ecfl / output_traj.shape[0]



def compute_ECFL_map_t(output_traj, map_path, t_gt, our_gt):
    '''
    :param output_traj: (# scenes, # samples, # frames, # coordinates) # all sample number, 20, 12, 2
    :param binary_navmaps: (# scenes, # height/y, # width/x)
        1 indicates navigable; 0 indicates non-navigable
    :return: avgECFL
    '''
    map_dir = 'D:\crowd\datasets\SDD\SDD_semantic_maps/test_masks'
    maps = {}
    for file in os.listdir(map_dir):
        m = imageio.imread(os.path.join(map_dir, file)).astype(float)
        m[np.argwhere(m == 1)[:, 0], np.argwhere(m == 1)[:, 1]] = 1
        m[np.argwhere(m == 2)[:, 0], np.argwhere(m == 2)[:, 1]] = 1
        m[np.argwhere(m == 3)[:, 0], np.argwhere(m == 3)[:, 1]] = 0
        m[np.argwhere(m == 4)[:, 0], np.argwhere(m == 4)[:, 1]] = 0
        m[np.argwhere(m == 5)[:, 0], np.argwhere(m == 5)[:, 1]] = 0
        maps.update({file.split('.')[0]: m})

    ecfl = 0.0
    for i in range(output_traj.shape[0]):
        idx = np.where(np.sqrt(((t_gt[i, 0] - our_gt[:, 0]) ** 2).sum(2)).mean(1) < 0.001)[0]
        if len(idx) >1:
            print(idx)
            print(t_gt[i, 0] - our_gt[idx, 0])
            print('----------------------')

        idx = idx[0]
        binary_navmaps = maps[map_path[idx]]

        for k in range(output_traj.shape[1]):
            collided = False
            wc = output_traj[i,k]

            # plt.imshow(binary_navmaps)
            # plt.scatter(wc[:, 0], wc[:, 1], c='r', s=2)
            for t in range(output_traj.shape[2]):
                pos = wc[t].astype(int)
                if pos[1] < 0 or pos[1] >= binary_navmaps.shape[0] or pos[0] < 0 or pos[0] >= binary_navmaps.shape[1]:
                    collided = True
                    break
                if binary_navmaps[pos[1], pos[0]] == 0:
                    collided = True
                    break

            if not collided:
                ecfl += 1.0 / output_traj.shape[1]

    return ecfl / output_traj.shape[0]


def compute_ECFL_t(output_traj, binary_navmaps, local_homo, t_gt, our_gt):
    '''
    :param output_traj: (# scenes, # samples, # frames, # coordinates) # all sample number, 20, 12, 2
    :param binary_navmaps: (# scenes, # height/y, # width/x)
        1 indicates navigable; 0 indicates non-navigable
    :return: avgECFL
    '''

    ecfl = 0.0
    for i in range(output_traj.shape[0]):
        idx = np.where(np.sqrt(((t_gt[i, 0] - our_gt[:, 0]) ** 2).sum(2)).mean(1) < 0.001)[0]
        if len(idx) >1:
            print(idx)
            print(t_gt[i, 0] - our_gt[idx, 0])
            print('----------------------')

        idx = idx[0]
        h = local_homo[idx]

        for k in range(output_traj.shape[1]):
            collided = False
            wc = output_traj[i,k]
            # wc = t_gt[i,0]

            all_pixel_local = np.matmul(np.concatenate([wc, np.ones((len(wc), 1))], axis=1),
                                        np.linalg.pinv(np.transpose(h)))
            all_pixel_local /= np.expand_dims(all_pixel_local[:, 2], 1)
            all_pixel_local = np.round(all_pixel_local).astype(int)[:, :2]

            # plt.imshow(binary_navmaps[i])
            # plt.scatter(all_pixel_local[:, 0], all_pixel_local[:, 1], c='r')

            for t in range(output_traj.shape[2]):
                pos = all_pixel_local[t]
                if pos[1] < 0 or pos[1] >= binary_navmaps[i].shape[1] or pos[0] < 0 or pos[0] >= binary_navmaps[i].shape[0]:
                    collided = True
                    break
                if binary_navmaps[i, pos[0], pos[1]] == 0:
                    collided = True
                    break

            if not collided:
                ecfl += 1.0 / output_traj.shape[1]

    return ecfl / output_traj.shape[0]

'''
From https://github.com/StanfordASL/Trajectron-plus-plus.git
'''
def compute_kde_nll(predicted_trajs, gt_traj, lower_bound):
    predicted_trajs = predicted_trajs.transpose(1,0,2,3)
    gt_traj = gt_traj[0]
    log_pdf_lower_bound = -20
    num_timesteps = predicted_trajs.shape[2]
    num_seq = predicted_trajs.shape[0]

    all_kde = []
    for i in range(num_seq):
        kde_ll = 0.
        for timestep in range(num_timesteps):
            try:
                kde = gaussian_kde(predicted_trajs[i, :, timestep].T)
                pdf = kde.logpdf(gt_traj[i, timestep].T)
                if lower_bound:
                    pdf = np.clip(pdf, a_min=log_pdf_lower_bound, a_max=None)
                kde_ll += pdf[0]
            except np.linalg.LinAlgError:
                print('nan')
                kde_ll = np.nan
        all_kde.append(-kde_ll/num_timesteps)

    return np.array(all_kde).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--file_path', default='D:\crowd\AgentFormer sdd k=20', type=str, help='pkl file dir' )
    parser.add_argument('--file_path', default='D:\crowd', type=str, help='pkl file dir' )
    # parser.add_argument('--file_path', default='D:\crowd\AgentFormer pathfinding k=20', type=str, help='pkl file dir' )
    parser.add_argument('--file_name', default='path_c_5', type=str, help='pkl file name with file extension')
    args = parser.parse_args()

    import pickle5
    with open(os.path.join(args.file_path, args.file_name + '.pkl'), 'rb') as handle:
        all_data = pickle5.load(handle) # (prediction, GT) where prediction.shape = gt.shape = (k, n, # future steps, 2)


    # with open('D:\crowd\AgentFormer pathfinding k=20/pathfinding_20_AF.pkl', 'rb') as f:
    # with open('C:/Users\Mihee\Documents\카카오톡 받은 파일/ynet_pathfinding_k20.pkl', 'rb') as f:

    ## ours-sdd-gt
    # print(compute_ECFL_map(np.expand_dims(all_data[1], 1), all_data[2]))
    ## ours-sdd-pred
    # print(compute_ECFL_map(all_data[0], all_data[2]))

    ## t++ sdd
    # with open(os.path.join('D:\crowd/sdd_c_5.pkl'), 'rb') as handle:
    #     all_data = pickle5.load(handle)
    # with open('D:\crowd/t_sdd20.pkl', 'rb') as f:
    # # with open('D:\crowd\AgentFormer sdd k=20/sdd_20_AF.pkl', 'rb') as f:
    #     tt = pickle5.load(f)
    # print(compute_ECFL_map_t(tt[0], all_data[2], tt[1].transpose(1,0,2,3)[:,:1], np.expand_dims(all_data[1], 1)))

    '''
    ## t++ path
    with open(os.path.join('D:\crowd/path_c_20.pkl'), 'rb') as handle:
        all_data = pickle5.load(handle)
    with open('D:\crowd/noprior_path_20.pkl', 'rb') as f:
    # with open('C:\dataset/t++\experiments\pedestrians/t_path_20.pkl', 'rb') as f:
    # with open('D:\crowd\AgentFormer pathfinding k=20/pathfinding_20_AF.pkl', 'rb') as f:
        tt = pickle5.load(f)
    print(compute_ECFL_t(tt[0].transpose(1,0,2,3), all_data[2], all_data[3], tt[1].transpose(1,0,2,3)[:,:1], np.expand_dims(all_data[1], 1)))
    '''

    ## ours-path-pred
    print("pred: ", compute_ECFL(all_data[0], all_data[2], all_data[3]))
    ## ours-path-gt
    print("gt: ", compute_ECFL(np.expand_dims(all_data[1], 1), all_data[2], all_data[3]))

    all_data[0] = all_data[0].transpose(1,0,2,3)
    all_data[1] = np.expand_dims(all_data[1], 0).repeat(all_data[0].shape[0], 0)

    print('>>> file name: ', args.file_name)
    print('=== ADE min / avg / std ===' )
    print(compute_ade(all_data[0], all_data[1]))
    print('=== FDE min / avg / std ===' )
    print(compute_fde(all_data[0][:,:,-1,:], all_data[1][:,:,-1,:]))
    print('=== NLL without lower bound ===' )
    print(compute_kde_nll(all_data[0], all_data[1], False))
    print('=== NLL with lower bound ===' )
    print(compute_kde_nll(all_data[0], all_data[1], True))
