import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from scipy.stats import gaussian_kde
import argparse
import os
import pickle
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
    parser.add_argument('--file_name', default='path_c_20', type=str, help='pkl file name with file extension')
    args = parser.parse_args()

    import pickle5
    with open(os.path.join(args.file_path, args.file_name + '.pkl'), 'rb') as handle:
        all_data = pickle5.load(handle) # (prediction, GT) where prediction.shape = gt.shape = (k, n, # future steps, 2)

    print(compute_ECFL(all_data[0], all_data[2], all_data[3]))
    print(compute_ECFL(np.expand_dims(all_data[1], 1), all_data[2], all_data[3]))

    print('>>> file name: ', args.file_name)
    print('=== ADE min / avg / std ===' )
    print(compute_ade(all_data[0], all_data[1]))
    print('=== FDE min / avg / std ===' )
    print(compute_fde(all_data[0][:,:,-1,:], all_data[1][:,:,-1,:]))
    print('=== NLL without lower bound ===' )
    print(compute_kde_nll(all_data[0], all_data[1], False))
    print('=== NLL with lower bound ===' )
    print(compute_kde_nll(all_data[0], all_data[1], True))
