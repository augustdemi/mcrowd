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

'''
From https://github.com/StanfordASL/Trajectron-plus-plus.git
'''
def compute_kde_nll(predicted_trajs, gt_traj):
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
                pdf = np.clip(kde.logpdf(gt_traj[i, timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
                kde_ll += pdf
            except np.linalg.LinAlgError:
                print('nan')
                kde_ll = np.nan
        all_kde.append(-kde_ll/num_timesteps)

    return np.array(all_kde).mean()


def compute_obs_violations(predicted_trajs, map):
    obs_map = map.data

    interp_obs_map = RectBivariateSpline(range(obs_map.shape[1]),
                                         range(obs_map.shape[0]),
                                         binary_dilation(obs_map.T, iterations=4),
                                         kx=1, ky=1)

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False)
    traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=1) > 0, dtype=float)

    return num_viol_trajs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default='../', type=str, help='predicted result path' )
    parser.add_argument('--file_name', default='sdd', type=str, help='modelname_datasetname_k such as ynet_sdd_5')
    args = parser.parse_args()

    with open(os.path.join(args.file_path, args.file_name + '.pkl'), 'rb') as handle:
        all_data = pickle.load(handle)

    print('>>> file name: ', args.file_name)
    print('=== ADE min / avg / std ===' )
    print(compute_ade(all_data[0], all_data[1]))
    print('=== FDE min / avg / std ===' )
    print(compute_fde(all_data[0][:,:,-1,:], all_data[1][:,:,-1,:]))
    print('=== NLL ===' )
    print(compute_kde_nll(all_data[0], all_data[1]))
