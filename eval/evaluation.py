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
    parser.add_argument('--file_path', default='../res', type=str, help='pkl file dir' )
    parser.add_argument('--file_name', default='sdd_5', type=str, help='pkl file name with file extension')
    args = parser.parse_args()

    import pickle5
    with open(os.path.join(args.file_path, args.file_name + '.pkl'), 'rb') as handle:
        all_data = pickle5.load(handle) # (prediction, GT) where prediction.shape = gt.shape = (k, n, # future steps, 2)

    print('>>> file name: ', args.file_name)
    print('=== ADE min / avg / std ===' )
    print(compute_ade(all_data[0], all_data[1]))
    print('=== FDE min / avg / std ===' )
    print(compute_fde(all_data[0][:,:,-1,:], all_data[1][:,:,-1,:]))
    print('=== NLL without lower bound ===' )
    print(compute_kde_nll(all_data[0], all_data[1], False))
    print('=== NLL with lower bound ===' )
    print(compute_kde_nll(all_data[0], all_data[1], True))
