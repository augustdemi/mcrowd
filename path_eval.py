import os
from matplotlib.animation import FuncAnimation

import torch.optim as optim
# -----------------------------------------------------------------------------#
from utils import DataGather, mkdirs, grid2gif2, apply_poe, sample_gaussian, sample_gumbel_softmax
from model import *
from loss import kl_two_gaussian, displacement_error, final_displacement_error
from data.loader import data_loader
import imageio
from scipy import ndimage

import matplotlib.pyplot as plt
from torch.distributions import RelaxedOneHotCategorical as concrete
from torch.distributions import OneHotCategorical as discrete
from torch.distributions import kl_divergence
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
# from model_map_ae import Decoder as Map_Decoder
from unet.probabilistic_unet import ProbabilisticUnet
from unet.unet import Unet
import numpy as np
import visdom
import torch.nn.functional as F
from unet.utils import init_weights


###############################################################################

def integrate_samples(v, p_0, dt=1):
    """
    Integrates deterministic samples of velocity.

    :param v: Velocity samples
    :return: Position samples
    """
    v=v.permute(1, 0, 2)
    abs_traj = torch.cumsum(v, dim=1) * dt + p_0.unsqueeze(1)
    return  abs_traj.permute((1, 0, 2))

# def recon_loss_with_logit(input, target):
#     nn.BCEWithLogitsLoss(size_average = False, reduce=False, reduction=None)

def make_one_heatmap(local_ic):
    heat_map_traj = np.zeros((160, 160))
    heat_map_traj[local_ic[0], local_ic[1]] = 1
    # as Y-net used variance 4 for the GT heatmap representation.
    heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)

    return heat_map_traj


class Solver(object):

    ####
    def __init__(self, args):

        self.args = args
        args.num_sg = args.load_e

        self.name = '%s_lr_%s_a_%s_r_%s' % \
                    (args.dataset_name, args.lr_VAE, args.alpha, args.gamma)
        # self.name = 'sg_enc_block_1_fcomb_block_2_wD_10_lr_0.001_lg_klw_1_a_0.25_r_2.0_fb_2.0_anneal_e_10_load_e_1'

        # to be appended by run_id

        # self.use_cuda = args.cuda and torch.cuda.is_available()
        self.fb = args.fb
        self.anneal_epoch = args.anneal_epoch
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.device = args.device
        self.temp=1.99
        self.dt=0.4
        self.eps=1e-9
        self.ll_prior_w =args.ll_prior_w
        self.sg_idx = np.array(range(12))
        self.sg_idx = np.flip(11-self.sg_idx[::(12//args.num_sg)])
        print('>>>>>>>>>> sg :', self.sg_idx)
        self.no_convs_fcomb = args.no_convs_fcomb
        self.no_convs_per_block = args.no_convs_per_block

        self.kl_weight=args.kl_weight
        self.lg_kl_weight=args.lg_kl_weight

        self.max_iter = int(args.max_iter)


        # do it every specified iters
        self.print_iter = args.print_iter
        self.ckpt_save_iter = args.ckpt_save_iter
        self.output_save_iter = args.output_save_iter

        # data info
        self.dataset_dir = args.dataset_dir
        self.dataset_name = args.dataset_name

        # self.N = self.latent_values.shape[0]
        # self.eval_metrics_iter = args.eval_metrics_iter

        # networks and optimizers
        self.batch_size = args.batch_size
        self.zS_dim = args.zS_dim
        self.w_dim = args.w_dim
        self.lr_VAE = args.lr_VAE
        self.beta1_VAE = args.beta1_VAE
        self.beta2_VAE = args.beta2_VAE
        print(args.desc)

        # create dirs: "records", "ckpts", "outputs" (if not exist)
        mkdirs("records");
        mkdirs("ckpts");
        mkdirs("outputs")

        # set run id
        if args.run_id < 0:  # create a new id
            k = 0
            rfname = os.path.join("records", self.name + '_run_0.txt')
            while os.path.exists(rfname):
                k += 1
                rfname = os.path.join("records", self.name + '_run_%d.txt' % k)
            self.run_id = k
        else:  # user-provided id
            self.run_id = args.run_id

        # finalize name
        self.name = self.name + '_run_' + str(self.run_id)

        # records (text file to store console outputs)
        self.record_file = 'records/%s.txt' % self.name

        # checkpoints
        self.ckpt_dir = os.path.join("ckpts", self.name)

        # outputs
        self.output_dir_recon = os.path.join("outputs", self.name + '_recon')
        # dir for reconstructed images
        self.output_dir_synth = os.path.join("outputs", self.name + '_synth')
        # dir for synthesized images
        self.output_dir_trvsl = os.path.join("outputs", self.name + '_trvsl')

        #### create a new model or load a previously saved model

        self.ckpt_load_iter = args.ckpt_load_iter

        self.obs_len = args.obs_len
        self.pred_len = args.pred_len
        self.num_layers = args.num_layers
        self.decoder_h_dim = args.decoder_h_dim



        # long_dtype, float_dtype = get_dtypes(args)

        if self.ckpt_load_iter != self.max_iter:
            print("Initializing train dataset")
            _, self.train_loader = data_loader(self.args, self.dataset_dir, 'train')
            print("Initializing val dataset")
            _, self.val_loader = data_loader(self.args, self.dataset_dir, 'val')

            print(
                'There are {} iterations per epoch'.format(len(self.train_loader.dataset) / args.batch_size)
            )
        print('...done')

        self.recon_loss_with_logit = nn.BCEWithLogitsLoss(size_average = False, reduce=False, reduction=None)



    def make_one_heatmap(self, local_ic):
        heat_map_traj = np.zeros((160, 160))
        heat_map_traj[local_ic[0], local_ic[1]] = 1
        # as Y-net used variance 4 for the GT heatmap representation.
        heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)

        return heat_map_traj




    def make_heatmap(self, local_ic, local_map):
        heatmaps = []
        for i in range(len(local_ic)):
            ohm = [local_map[i, 0]]

            heat_map_traj = np.zeros((160, 160))
            for t in range(self.obs_len):
                heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 1
                # as Y-net used variance 4 for the GT heatmap representation.
            heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
            ohm.append( heat_map_traj/heat_map_traj.sum())

            for t in (self.sg_idx + 8):
                heat_map_traj = np.zeros((160,160))
                heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 1
                # as Y-net used variance 4 for the GT heatmap representation.
                heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
                # plt.imshow(heat_map_traj)
                ohm.append(heat_map_traj)
            heatmaps.append(np.stack(ohm))
            '''
            heat_map_traj = np.zeros((160, 160))
            # for t in range(self.obs_len + self.pred_len):
            for t in [0,1,2,3,4,5,6,7,11,14,17]:
                heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 1
                # as Y-net used variance 4 for the GT heatmap representation.
            heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
            plt.imshow(heat_map_traj)
            '''
        heatmaps = torch.tensor(np.stack(heatmaps)).float().to(self.device)
        return heatmaps[:,:2], heatmaps[:,2:], heatmaps[:,-1].unsqueeze(1)


    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor



    def check_feat(self, data_loader):
        self.set_mode(train=False)

        with torch.no_grad():
            b = 0
            for batch in data_loader:
                b += 1
                (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
                 obs_frames, pred_frames, map_path, inv_h_t,
                 local_map, local_ic, local_homo) = batch

                batch = data_loader.dataset.__getitem__(131)
                (obs_traj, fut_traj,
                 obs_frames, pred_frames, map_path, inv_h_t,
                 local_map, local_ic, local_homo, scale) = batch
                obs_traj = obs_traj.permute((2,0,1))
                fut_traj = fut_traj.permute((2,0,1))

                obs_traj_st = obs_traj.clone()
                # pos is stdized by mean = last obs step
                obs_traj_st[:, :, :2] = obs_traj_st[:, :, :2] - obs_traj_st[-1, :, :2]
                plt.imshow(local_map[0, 0])

                i=tmp_idx =0

                obs_heat_map, sg_heat_map, lg_heat_map = self.make_heatmap(local_ic, local_map)
                self.lg_cvae.forward(obs_heat_map, None, training=False)

                '''
                #paper image
                import cv2
                cv2.imwrite('D:\crowd\cvpr/fig/pathfinding local map', Image.fromarray(1-obs_heat_map[9,0].numpy()))
                cv2.imwrite('D:\crowd\cvpr/fig/pathfinding_local_map.png', 255-255*obs_heat_map[9,0].numpy())
                cv2.imwrite('D:\crowd\cvpr/fig/pathfinding_obs.png', 255*obs_heat_map[9,1].numpy())

                plt.tight_layout()
                fig = plt.imshow(sg_heat_map[9, 0])
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)


                env = 1-obs_heat_map[9,0].numpy()
                plt.imshow(np.stack([env * (1 - obs_heat_map[9,1].numpy() * 10),
                                     env * (1-lg_heat_map[9, 0].numpy() * 10), env], axis=2))

                # global map
                plt.tight_layout()
                fig = plt.imshow(cv2.imread('D:\crowd\datasets\Trajectories\map/43.png'))
                plt.scatter(obs_traj[:,9,0], obs_traj[:,9,1], s=1, c='b')
                plt.scatter(fut_traj[:,9,0], fut_traj[:,9,1], s=1, c='r')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)

                # local map
                plt.tight_layout()
                fig = plt.imshow(np.stack([255-255*obs_heat_map[9,0], 255-255*obs_heat_map[9,0], 255-255*obs_heat_map[9,0]],axis=2))
                plt.scatter(local_ic[9,:8,1], local_ic[9,:8,0], s=1, c='b')
                plt.scatter(local_ic[9,8:,1], local_ic[9,8:,0], s=1, c='r')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                '''

                ###################################################

                # self.lg_cvae.forward(obs_heat_map, None, training=False)

                zs = []
                for _ in range(10):
                    zs.append(self.lg_cvae.prior_latent_space.rsample())

                mm = []
                for k in range(5):
                    mm.append(F.sigmoid(self.lg_cvae.sample(self.lg_cvae.unet_enc_feat, zs[k])))
                # mm.append(F.sigmoid(self.lg_cvae.sample(self.lg_cvae.unet_enc_feat, self.lg_cvae.posterior_latent_space.rsample())))

                mmm = []
                for k in range(5,10):
                    mmm.append(F.sigmoid(self.lg_cvae.sample(self.lg_cvae.unet_enc_feat, zs[k])))
                # mm.append(F.sigmoid(self.lg_cvae.sample(self.lg_cvae.unet_enc_feat, self.lg_cvae.posterior_latent_space.rsample())))


                # env = local_map[i,0].detach().cpu().numpy()
                # heat_map_traj = torch.zeros((160,160))
                # for t in [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 19]:
                #     heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 20
                # heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2) *10


                env = 1-local_map[i,0]
                # for t in [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 19]:
                #     env[local_ic[i, t, 0], local_ic[i, t, 1]] = 0

                heat_map_traj = torch.zeros((160,160))
                for t in [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 19]:
                    heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 20
                heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=1)


                fig = plt.figure(figsize=(12, 10))
                fig.tight_layout()
                for k in range(10):
                    ax = fig.add_subplot(4, 5, k + 1)
                    ax.set_title('prior' + str(k % 5 + 1))
                    if k < 5:
                        a = mm[k][i, 0].detach().cpu().numpy().copy()
                        ax.imshow(np.stack([env*(1-heat_map_traj), env * (1-a *5),  env],axis=2))
                    else:
                        ax.imshow(mm[k % 5][i, 0])

                for k in range(10):
                    ax = fig.add_subplot(4, 5, k + 11)
                    ax.set_title('prior' + str(k % 5 + 6))
                    if k < 5:
                        a = mmm[k][i, 0].detach().cpu().numpy().copy()
                        ax.imshow(np.stack([env*(1-heat_map_traj), env * (1-a * 5),  env],axis=2))
                        # ax.imshow(np.stack([1-env, 1-heat_map_traj, 1 - mmm[k][i, 0] / (0.1*mmm[k][i, 0].max())],axis=2))
                    else:
                        ax.imshow(mmm[k % 5][i, 0])


                ####################### SG ############################

                pred_lg_prior = mm[-2]

                pred_lg_ics = []
                for k in range(lg_heat_map.shape[0]):
                    pred_lg_ic = []
                    for heat_map in pred_lg_prior[k]:
                        pred_lg_ic.append((heat_map == torch.max(heat_map)).nonzero()[0])
                    pred_lg_ic = torch.stack(pred_lg_ic).float()
                    pred_lg_ics.append(pred_lg_ic)

                pred_lg_heat_from_ic = []
                for coord in pred_lg_ics:
                    heat_map_traj = np.zeros((160, 160))
                    heat_map_traj[int(coord[0,0]), int(coord[0,1])] = 1
                    heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
                    pred_lg_heat_from_ic.append(heat_map_traj)
                pred_lg_heat_from_ic = torch.tensor(np.stack(pred_lg_heat_from_ic)).unsqueeze(1).float().to(self.device)

                pred_sg = F.sigmoid(
                    self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat_from_ic], dim=1)))


                env = 1-local_map[i,0]

                heat_map_traj = torch.zeros((160,160))
                for t in [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 19]:
                    heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 20
                heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=1)


                fig = plt.figure(figsize=(10, 5))

                for k in range(6):
                    m = pred_sg[i][k % 3].detach().cpu().numpy().copy()
                    ax = fig.add_subplot(2,3,k+1)
                    if k <3:
                        ax.set_title('sg' + str(k + 1))
                        ax.imshow(np.stack([env * (1 - heat_map_traj), env * (1 - m * 5), env], axis=2))
                    else:
                        ax.imshow(m)


                #################### GIF #################### GIF

                pred_lg_wcs = []
                pred_sg_wcs = []
                traj_num=1
                lg_num=20
                for _ in range(lg_num):
                    # -------- long term goal --------
                    pred_lg_heat = F.sigmoid(self.lg_cvae.sample(testing=True))
                # for pred_lg_heat in mmm:
                    pred_lg_wc = []
                    pred_lg_ics = []
                    for i in range(len(obs_heat_map)):
                        pred_lg_ic = []
                        for heat_map in pred_lg_heat[i]:
                            pred_lg_ic.append((heat_map == torch.max(heat_map)).nonzero()[0])
                        pred_lg_ic = torch.stack(pred_lg_ic).float()
                        pred_lg_ics.append(pred_lg_ic)

                        # ((local_ic[0,[11,15,19]] - pred_sg_ic) ** 2).sum(1).mean()
                        back_wc = torch.matmul(
                            torch.cat([pred_lg_ic, torch.ones((len(pred_lg_ic), 1)).to(self.device)], dim=1),
                            torch.transpose(local_homo[i], 1, 0))
                        pred_lg_wc.append(back_wc[0, :2] / back_wc[0, 2])
                        # ((back_wc - fut_traj[[3, 7, 11], 0, :2]) ** 2).sum(1).mean()
                    pred_lg_wc = torch.stack(pred_lg_wc)
                    pred_lg_wcs.append(pred_lg_wc)

                    # -------- short term goal --------
                    pred_lg_heat_from_ic = []
                    for coord in pred_lg_ics:
                        heat_map_traj = np.zeros((160, 160))
                        heat_map_traj[int(coord[0, 0]), int(coord[0, 1])] = 1
                        heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
                        pred_lg_heat_from_ic.append(heat_map_traj)
                    pred_lg_heat_from_ic = torch.tensor(np.stack(pred_lg_heat_from_ic)).unsqueeze(1).float().to(
                        self.device)

                    pred_sg_heat = F.sigmoid(
                        self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat_from_ic], dim=1)))
                    # pred_sg_heat = F.sigmoid(self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat], dim=1)))

                    pred_sg_wc = []
                    for s in range(len(obs_heat_map)):
                        pred_sg_ic = []
                        for heat_map in pred_sg_heat[s]:
                            pred_sg_ic.append((heat_map == torch.max(heat_map)).nonzero()[0])
                        pred_sg_ic = torch.stack(pred_sg_ic).float()

                        # ((local_ic[0,[11,15,19]] - pred_sg_ic) ** 2).sum(1).mean()
                        back_wc = torch.matmul(
                            torch.cat([pred_sg_ic, torch.ones((len(pred_sg_ic), 1)).to(self.device)], dim=1),
                            torch.transpose(local_homo[i], 1, 0))
                        back_wc /= back_wc[:, 2].unsqueeze(1)
                        pred_sg_wc.append(back_wc[:, :2])
                        # ((back_wc - fut_traj[[3, 7, 11], 0, :2]) ** 2).sum(1).mean()
                    pred_sg_wc = torch.stack(pred_sg_wc)
                    pred_sg_wcs.append(pred_sg_wc)

                ##### trajectories per long&short goal ####
                # -------- trajectories --------
                multi_sample_pred = []

                # zero_map_feat = self.lg_cvae.unet_enc_feat.clone()

                (hx, mux, log_varx) \
                    = self.encoderMx(obs_traj_st, seq_start_end, self.lg_cvae.unet_enc_feat, local_homo)

                p_dist = Normal(mux, torch.sqrt(torch.exp(log_varx)))
                z_priors = []
                for _ in range(traj_num):
                    z_priors.append(p_dist.sample())

                for pred_sg_wc in pred_sg_wcs:
                    for z_prior in z_priors:
                        # -------- trajectories --------
                        # NO TF, pred_goals, z~prior
                        fut_rel_pos_dist_prior = self.decoderMy(
                            obs_traj_st[-1],
                            obs_traj[-1, :, :2],
                            hx,
                            z_prior,
                            pred_sg_wc,  # goal
                            self.sg_idx
                        )
                        multi_sample_pred.append(fut_rel_pos_dist_prior.rsample())

                ## pixel data
                pred_data = []
                for pred in multi_sample_pred:
                    pred_fut_traj = integrate_samples(pred, obs_traj[-1, :, :2],
                                                      dt=self.dt)
                    one_ped = i

                    pred_real = pred_fut_traj[:, one_ped].numpy()
                    pred_pixel = np.concatenate([pred_real, np.ones((self.pred_len, 1))],
                                                axis=1)

                    pred_pixel = np.matmul(pred_pixel, np.linalg.inv(np.transpose(local_homo[one_ped])))
                    pred_pixel /= np.expand_dims(pred_pixel[:, 2], 1)
                    pred_data.append(np.concatenate([local_ic[i,:8], pred_pixel[:,:2]], 0))

                pred_data = np.expand_dims(np.stack(pred_data),1)


                #---------- plot gif

                env = 1-local_map[i,0]
                env = np.stack([env, env, env], axis=2)

                def init():
                    ax.imshow(env)

                def update_dot(num_t):
                    print(num_t)
                    ax.imshow(env)
                    for j in range(len(pred_data)):
                        print(j, i)
                        ln_pred[j].set_data(pred_data[j, i, :num_t, 1],
                                                   pred_data[j, i, :num_t, 0])
                    ln_gt.set_data(local_ic[i, :num_t, 1], local_ic[i, :num_t, 0])

                fig, ax = plt.subplots()
                ax.set_title(str(i), fontsize=9)
                fig.tight_layout()
                colors = ['r', 'g', 'b', 'm', 'c', 'k', 'w', 'k']

                # ln_gt.append(ax.plot([], [], colors[i % len(colors)] + '--', linewidth=1)[0])
                # ln_gt.append(ax.scatter([], [], c=colors[i % len(colors)], s=2))

                ln_pred = []
                for _ in range(len(pred_data)):
                    ln_pred.append(
                        ax.plot([], [], 'r', alpha=0.5, linewidth=1)[0])

                ln_gt = ax.plot([], [], 'b--', linewidth=1)[0]



                ani = FuncAnimation(fig, update_dot, frames=20, interval=1, init_func=init())

                # writer = PillowWriter(fps=3000)
                gif_path = 'D:\crowd\datasets\Trajectories'
                ani.save(gif_path + "/" "path_find_agent" + str(i) + ".gif", fps=4)

                # ====================================================================================================
                # ==================================================for report ==================================================
                # ====================================================================================================


                ###################### OURS ########################################################################################


                import matplotlib.patheffects as pe
                import seaborn as sns
                from matplotlib.offsetbox import OffsetImage, AnnotationBbox
                ####### plot 20 trajs ############
                i = tmp_idx
                env = np.expand_dims((1 - local_map[i][0]), 2).repeat(3, 2)

                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(1, 1, 1)
                plt.tight_layout()
                # env = OffsetImage(env, zoom=0.06)
                # env = AnnotationBbox(env, (0.5, 0.5),
                #                      bboxprops=dict(edgecolor='red'))
                #
                # ax.add_artist(env)
                ax.imshow(env)
                ax.axis('off')

                for t in range(self.obs_len, self.obs_len + self.pred_len):
                    # sns.kdeplot(pred_data[:, 0, t, 1], pred_data[:, 0, t, 0],shade=True)
                    sns.kdeplot(pred_data[:, 0, t, 1], pred_data[:, 0, t, 0], bw=None,
                                ax=ax, shade=True, shade_lowest=False,
                                color='r', zorder=600, alpha=0.6, legend=True)

                # ax.scatter(local_ic[0,:,1], local_ic[0,:,0], s=5, c='b')

                for t in range(20):
                    if t == 0:
                        ax.plot(pred_data[t, 0, 8:, 1], pred_data[t, 0, 8:, 0], 'r--', linewidth=2,
                                zorder=650, c='firebrick',
                                path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()],
                                label='Predicted future')
                    else:
                        ax.plot(pred_data[t, 0, 8:, 1], pred_data[t, 0, 8:, 0], 'r--', linewidth=2,
                                zorder=650, c='firebrick',
                                path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])

                ax.plot(local_ic[0, 8:, 1], local_ic[0, 8:, 0],
                        'r--o',
                        c='darkorange',
                        linewidth=2,
                        markersize=2,
                        zorder=650,
                        path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()], label='GT future')
                ax.plot(local_ic[0, :8, 1], local_ic[0, :8, 0],
                        'b--o',
                        c='royalblue',
                        linewidth=2,
                        markersize=2,
                        zorder=650,
                        path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()], label='GT past')

                ax.legend()

                # def frame_image(img, frame_width):
                #     b = frame_width  # border size in pixel
                #     ny, nx = img.shape[0], img.shape[1]  # resolution / number of pixels in x and y
                #     if img.ndim == 3:  # rgb or rgba array
                #         framed_img = np.zeros((b + ny + b, b + nx + b, img.shape[2]))
                #     elif img.ndim == 2:  # grayscale image
                #         framed_img = np.zeros((b + ny + b, b + nx + b))
                #     framed_img[b:-b, b:-b] = img
                #     return framed_img

                ################### LG ##############################
                # ------- plot -----------
                idx_list = range(0,3)

                fig = plt.figure(figsize=(12, 10))
                for h in range(3):
                    idx = idx_list[h]
                    ax = fig.add_subplot(3, 4, 4 * h + 1)
                    plt.tight_layout()
                    ax.imshow(env)
                    ax.axis('off')
                    # ax.scatter(local_ic[0,:4,1], local_ic[0,:4,0], c='b', s=2, alpha=0.7)
                    # ax.scatter(local_ic[0,4:,1], local_ic[0,4:,0], c='r', s=2, alpha=0.7)
                    # ax.scatter(local_ic[0,-1,1], local_ic[0,-1,0], c='r', s=18, marker='x', alpha=0.7)

                    ax.plot(local_ic[0, :8, 1], local_ic[0, :8, 0],
                            'b--o',
                            c='royalblue',
                            linewidth=1,
                            markersize=1.5,
                            zorder=500,
                            path_effects=[pe.Stroke(linewidth=1, foreground='k'), pe.Normal()], label='GT past')
                    ax.plot(local_ic[0, 8:, 1], local_ic[0, 8:, 0],
                            'r--o',
                            c='darkorange',
                            linewidth=1,
                            markersize=1.5,
                            zorder=500,
                            path_effects=[pe.Stroke(linewidth=1, foreground='k'), pe.Normal()], label='GT future')

                    ax.plot(local_ic[0, -1, 1], local_ic[0, -1, 0],
                            'r--x',
                            c='darkorange',
                            linewidth=1,
                            markersize=4,
                            zorder=500,
                            path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()], label='GT past')

                    a = mm[idx][i, 0]

                    c = a / a.max()
                    d=np.stack([1-c, np.ones_like(c), np.ones_like(c)]).transpose(1,2,0)
                    ax.imshow(d, alpha=0.7)


                    # -------- short term goal --------

                    pred_lg_heat = mm[idx]
                    pred_lg_ics = []
                    for j in range(len(obs_heat_map)):
                        map_size = local_map[j][0].shape
                        pred_lg_ic = []
                        for heat_map in pred_lg_heat[j]:
                            argmax_idx = heat_map.argmax()
                            argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                            pred_lg_ic.append(argmax_idx)

                        pred_lg_ic = torch.tensor(pred_lg_ic).float().to(self.device)
                        pred_lg_ics.append(pred_lg_ic)

                    pred_lg_heat_from_ic = []
                    for lg_idx in range(len(pred_lg_ics)):
                        pred_lg_heat_from_ic.append(make_one_heatmap(pred_lg_ics[
                            lg_idx].detach().cpu().numpy().astype(int)[0]))
                    pred_lg_heat_from_ic = torch.tensor(np.stack(pred_lg_heat_from_ic)).unsqueeze(1).float().to(
                        self.device)
                    pred_sg = F.sigmoid(
                        self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat_from_ic], dim=1)))

                    m = pred_sg.detach().cpu().numpy().copy()[i]
                    for jj in range(3):
                        ax = fig.add_subplot(3, 4, 4 * h + 2 + jj)
                        plt.tight_layout()
                        ax.imshow(env)
                        ax.axis('off')
                        # ax.scatter(local_ic[0, :4, 1], local_ic[0, :4, 0], c='b', s=2, alpha=0.7)
                        # ax.scatter(local_ic[0, 4:, 1], local_ic[0, 4:, 0], c='r', s=2, alpha=0.7)
                        # ax.scatter(local_ic[0, self.sg_idx[jj] + self.obs_len, 1], local_ic[0,  self.sg_idx[jj] + self.obs_len, 0], c='r', s=18, marker='x', alpha=0.7)
                        ax.plot(local_ic[0, :8, 1], local_ic[0, :8, 0],
                                'b--o',
                                c='royalblue',
                                linewidth=1,
                                markersize=1.5,
                                zorder=500,
                                path_effects=[pe.Stroke(linewidth=1, foreground='k'), pe.Normal()], label='GT past')
                        ax.plot(local_ic[0, 8:, 1], local_ic[0, 8:, 0],
                                'r--o',
                                c='darkorange',
                                linewidth=1,
                                markersize=1.5,
                                zorder=500,
                                path_effects=[pe.Stroke(linewidth=1, foreground='k'), pe.Normal()], label='GT future')

                        ax.plot(local_ic[0, self.sg_idx[jj] + self.obs_len, 1],
                                local_ic[0, self.sg_idx[jj] + self.obs_len, 0],
                                'r--x',
                                c='darkorange',
                                linewidth=1,
                                markersize=4,
                                zorder=500,
                                path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()], label='GT past')
                        c = m[jj] / m[jj].max()
                        d = np.stack([1 - c, np.ones_like(c), np.ones_like(c)]).transpose(1, 2, 0)
                        ax.imshow(d, alpha=0.7)

                        ##@@@@@@@@@@@@@@@@@@@@@@@@
                        zs = []
                        for _ in range(20):
                            zs.append(self.lg_cvae.prior_latent_space.rsample())

                        mm = []
                        for k in range(20):
                            mm.append(F.sigmoid(self.lg_cvae.sample(self.lg_cvae.unet_enc_feat, zs[k])))
                        ##@@@@@@@@@@@@@@@@@@@@@@@@

                        # ==============================================================================================
                        # ===========================================baseline===================================================
                        # ==============================================================================================

                        ######################
                        # T++
                        import pickle5
                        # with open('C:\dataset/t++\experiments\pedestrians/t_path_20.pkl', 'rb') as f:
                        with open('D:\crowd\AgentFormer pathfinding k=20/pathfinding_20_AF.pkl', 'rb') as f:
                        # with open('C:/Users\Mihee\Documents\카카오톡 받은 파일/ynet_pathfinding_k20.pkl', 'rb') as f:
                            aa = pickle5.load(f)
                        our_gt = fut_traj[:, 0, :2].numpy()
                        idx= np.where(((aa[1][0,:,0] - our_gt[0])**2).sum(1) <1)[0][1]
                        gt = aa[1][0, idx]
                        gt - our_gt
                        af_pred = aa[0][:, idx]

                        ## pixel data
                        af_pred_data = []
                        for pred_real in af_pred:
                            pred_pixel = np.concatenate([pred_real, np.ones((self.pred_len, 1))],
                                                        axis=1)

                            pred_pixel = np.matmul(pred_pixel, np.linalg.inv(np.transpose(local_homo[tmp_idx])))
                            pred_pixel /= np.expand_dims(pred_pixel[:, 2], 1)
                            af_pred_data.append(np.concatenate([local_ic[i, :8], pred_pixel[:, :2]], 0))

                        af_pred_data = np.expand_dims(np.stack(af_pred_data), 1)  # (20, 1, 16, 2)

                        import matplotlib.patheffects as pe
                        import seaborn as sns
                        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
                        ####### plot 20 trajs ############
                        i = 0
                        env = np.expand_dims((1 - local_map[i][0]), 2).repeat(3, 2)

                        fig = plt.figure(figsize=(5, 5))
                        ax = fig.add_subplot(1, 1, 1)
                        plt.tight_layout()
                        # env = OffsetImage(env, zoom=0.06)
                        # env = AnnotationBbox(env, (0.5, 0.5),
                        #                      bboxprops=dict(edgecolor='red'))
                        #
                        # ax.add_artist(env)
                        ax.imshow(env)
                        ax.axis('off')
                        for t in range(self.obs_len, self.obs_len + self.pred_len):
                            # sns.kdeplot(pred_data[:, 0, t, 1], pred_data[:, 0, t, 0],shade=True)
                            sns.kdeplot(af_pred_data[:, 0, t, 1], af_pred_data[:, 0, t, 0], bw=None,
                                        ax=ax, shade=True, shade_lowest=False,
                                        color='r', zorder=600, alpha=0.6, legend=True)

                        # ax.scatter(local_ic[0,:,1], local_ic[0,:,0], s=5, c='b')

                        for t in range(20):
                            if t == 0:
                                ax.plot(af_pred_data[t, 0, 8:, 1], af_pred_data[t, 0, 8:, 0], 'r--', linewidth=2,
                                        zorder=650, c='firebrick',
                                        path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()],
                                        label='Predicted future')
                            else:
                                ax.plot(af_pred_data[t, 0, 8:, 1], af_pred_data[t, 0, 8:, 0], 'r--', linewidth=2,
                                        zorder=650, c='firebrick',
                                        path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])

                        ax.plot(local_ic[0, 8:, 1], local_ic[0, 8:, 0],
                                'r--o',
                                # c='darkorange',
                                c='darkorange',
                                linewidth=2,
                                markersize=2,
                                zorder=650,
                                path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()], label='GT future')
                        ax.plot(local_ic[0, :8, 1], local_ic[0, :8, 0],
                                'b--o',
                                c='royalblue',
                                linewidth=2,
                                markersize=2,
                                zorder=650,
                                path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()], label='GT past')

                        ax.legend()

                        # ==============================================================================================
                        # ==============================================================================================


                        ######################
                        # AF
                        import pickle5
                        with open('C:\dataset\AgentFormer nuscenes k=10\AgentFormer nuscenes k=10/nuscenes_10.pkl',
                                  'rb') as f:
                            aa = pickle5.load(f)
                        gt = aa[1][0, 818]
                        af_pred = aa[0][:, 818]
                        our_gt = fut_traj[:, 0, :2]

                        ## pixel data
                        af_pred_data = []
                        for pred_real in af_pred:
                            pred_pixel = np.concatenate([pred_real, np.ones((self.pred_len, 1))],
                                                        axis=1)

                            pred_pixel = np.matmul(pred_pixel, np.linalg.inv(np.transpose(local_homo[tmp_idx])))
                            pred_pixel /= np.expand_dims(pred_pixel[:, 2], 1)
                            af_pred_data.append(np.concatenate([local_ic[i, :4], pred_pixel[:, :2]], 0))

                        af_pred_data = np.expand_dims(np.stack(af_pred_data), 1)  # (20, 1, 16, 2)

                        import matplotlib.patheffects as pe
                        import seaborn as sns
                        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
                        ####### plot 20 trajs ############
                        i = tmp_idx
                        env = np.expand_dims((1 - local_map[i]), 2).repeat(3, 2)
                        # env = 1-local_map[i]

                        fig = plt.figure(figsize=(5, 5))
                        ax = fig.add_subplot(1, 1, 1)
                        plt.tight_layout()
                        # env = OffsetImage(env, zoom=0.06)
                        # env = AnnotationBbox(env, (0.5, 0.5),
                        #                      bboxprops=dict(edgecolor='red'))
                        #
                        # ax.add_artist(env)
                        ax.imshow(env)
                        ax.axis('off')

                        for t in range(self.obs_len, self.obs_len + self.pred_len):
                            # sns.kdeplot(pred_data[:, 0, t, 1], pred_data[:, 0, t, 0],shade=True)
                            sns.kdeplot(af_pred_data[:, 0, t, 1], af_pred_data[:, 0, t, 0],
                                        ax=ax, shade=True, shade_lowest=False,
                                        color='r', zorder=600, alpha=0.3, legend=True)

                        # ax.scatter(local_ic[0,:,1], local_ic[0,:,0], s=5, c='b')

                        for t in range(10):
                            if t == 0:
                                ax.plot(af_pred_data[t, 0, 4:, 1], af_pred_data[t, 0, 4:, 0], 'r--', linewidth=2,
                                        zorder=650, c='firebrick',
                                        path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()],
                                        label='Predicted future')
                            else:
                                ax.plot(af_pred_data[t, 0, 4:, 1], af_pred_data[t, 0, 4:, 0], 'r--', linewidth=2,
                                        zorder=650, c='firebrick',
                                        path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])

                        ax.plot(local_ic[0, 4:, 1], local_ic[0, 4:, 0],
                                'r--o',
                                c='darkorange',
                                linewidth=2,
                                markersize=2,
                                zorder=650,
                                path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()], label='GT future')
                        ax.plot(local_ic[0, :4, 1], local_ic[0, :4, 0],
                                'b--o',
                                c='royalblue',
                                linewidth=2,
                                markersize=2,
                                zorder=650,
                                path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()], label='GT past')

                        ax.legend()



    def all_evaluation(self, data_loader, lg_num=5, traj_num=4, generate_heat=True):
        self.set_mode(train=False)
        total_traj = 0

        all_ade =[]
        all_fde =[]
        sg_ade=[]
        lg_fde=[]
        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
                 obs_frames, pred_frames, map_path, inv_h_t,
                 local_map, local_ic, local_homo) = batch
                batch_size = obs_traj.size(1)
                total_traj += fut_traj.size(1)

                obs_heat_map, _, _= self.make_heatmap(local_ic, local_map)

                self.lg_cvae.forward(obs_heat_map, None, training=False)
                fut_rel_pos_dists = []
                pred_lg_wcs = []
                pred_sg_wcs = []

                ####### long term goals and the corresponding (deterministic) short term goals ########
                w_priors = []
                for _ in range(lg_num):
                    w_priors.append(self.lg_cvae.prior_latent_space.sample())

                for w_prior in w_priors:
                    # -------- long term goal --------
                    pred_lg_heat = F.sigmoid(self.lg_cvae.sample(self.lg_cvae.unet_enc_feat, w_prior))

                    pred_lg_wc = []
                    pred_lg_ics = []
                    for i in range(batch_size):
                        map_size = local_map[i][0].shape
                        pred_lg_ic = []
                        for heat_map in pred_lg_heat[i]:
                            argmax_idx = heat_map.argmax()
                            argmax_idx = [argmax_idx//map_size[0], argmax_idx%map_size[0]]
                            pred_lg_ic.append(argmax_idx)
                        pred_lg_ic = torch.tensor(pred_lg_ic).float().to(self.device)
                        pred_lg_ics.append(pred_lg_ic)

                        # ((local_ic[0,[11,15,19]] - pred_sg_ic) ** 2).sum(1).mean()
                        back_wc = torch.matmul(
                            torch.cat([pred_lg_ic, torch.ones((len(pred_lg_ic), 1)).to(self.device)], dim=1),
                            torch.transpose(local_homo[i].float().to(self.device), 1, 0))
                        pred_lg_wc.append(back_wc[0, :2] / back_wc[0, 2])
                        # ((back_wc - fut_traj[[3, 7, 11], 0, :2]) ** 2).sum(1).mean()
                    pred_lg_wc = torch.stack(pred_lg_wc)
                    pred_lg_wcs.append(pred_lg_wc)

                    # -------- short term goal --------
                    # obs_lg_heat = torch.cat([obs_heat_map, pred_lg_heat[:, -1].unsqueeze(1)], dim=1)

                    if generate_heat:
                        # -------- short term goal --------
                        pred_lg_heat_from_ic = []
                        for coord in pred_lg_ics:
                            heat_map_traj = np.zeros((160, 160))
                            heat_map_traj[int(coord[0, 0]), int(coord[0, 1])] = 1
                            heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
                            pred_lg_heat_from_ic.append(heat_map_traj)
                        pred_lg_heat_from_ic = torch.tensor(np.stack(pred_lg_heat_from_ic)).unsqueeze(1).float().to(
                            self.device)

                        pred_sg_heat = F.sigmoid(self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat_from_ic], dim=1)))
                    else:
                        pred_sg_heat = F.sigmoid(self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat], dim=1)))


                    pred_sg_wc = []
                    for i in range(batch_size):
                        pred_sg_ic = []
                        for heat_map in pred_sg_heat[i]:
                            argmax_idx = heat_map.argmax()
                            argmax_idx = [argmax_idx//map_size[0], argmax_idx%map_size[0]]
                            pred_sg_ic.append(argmax_idx)
                        pred_sg_ic = torch.tensor(pred_sg_ic).float().to(self.device)

                        # ((local_ic[0,[11,15,19]] - pred_sg_ic) ** 2).sum(1).mean()
                        back_wc = torch.matmul(
                            torch.cat([pred_sg_ic, torch.ones((len(pred_sg_ic), 1)).to(self.device)], dim=1),
                            torch.transpose(local_homo[i].float().to(self.device), 1, 0))
                        back_wc /= back_wc[:, 2].unsqueeze(1)
                        pred_sg_wc.append(back_wc[:, :2])
                        # ((back_wc - fut_traj[[3, 7, 11], 0, :2]) ** 2).sum(1).mean()
                    pred_sg_wc = torch.stack(pred_sg_wc)
                    pred_sg_wcs.append(pred_sg_wc)

                ##### trajectories per long&short goal ####

                # -------- trajectories --------
                (hx, mux, log_varx) \
                    = self.encoderMx(obs_traj_st, seq_start_end, self.lg_cvae.unet_enc_feat, local_homo)

                p_dist = Normal(mux, torch.sqrt(torch.exp(log_varx)))
                z_priors = []
                for _ in range(traj_num):
                    z_priors.append(p_dist.sample())

                for pred_sg_wc in pred_sg_wcs:
                    for z_prior in z_priors:
                        # -------- trajectories --------
                        # NO TF, pred_goals, z~prior
                        fut_rel_pos_dist_prior = self.decoderMy(
                            obs_traj_st[-1],
                            obs_traj[-1, :, :2],
                            hx,
                            z_prior,
                            pred_sg_wc,  # goal
                            self.sg_idx
                        )
                        fut_rel_pos_dists.append(fut_rel_pos_dist_prior)


                ade, fde = [], []
                for dist in fut_rel_pos_dists:
                    pred_fut_traj=integrate_samples(dist.rsample(), obs_traj[-1, :, :2], dt=self.dt)
                    ade.append(displacement_error(
                        pred_fut_traj, fut_traj[:,:,:2], mode='raw'
                    ))
                    fde.append(final_displacement_error(
                        pred_fut_traj[-1], fut_traj[-1,:,:2], mode='raw'
                    ))
                all_ade.append(torch.stack(ade))
                all_fde.append(torch.stack(fde))
                sg_ade.append(torch.sqrt(((torch.stack(pred_sg_wcs).permute(0, 2, 1, 3)
                                           - fut_traj[list(self.sg_idx),:,:2].unsqueeze(0).repeat((lg_num,1,1,1)))**2).sum(-1)).sum(1)) # 20, 3, 4, 2
                lg_fde.append(torch.sqrt(((torch.stack(pred_lg_wcs)
                                           - fut_traj[-1,:,:2].unsqueeze(0).repeat((lg_num,1,1)))**2).sum(-1))) # 20, 3, 4, 2

            all_ade=torch.cat(all_ade, dim=1).cpu().numpy()
            all_fde=torch.cat(all_fde, dim=1).cpu().numpy()
            sg_ade=torch.cat(sg_ade, dim=1).cpu().numpy()
            lg_fde=torch.cat(lg_fde, dim=1).cpu().numpy() # all batches are concatenated

            ade_min = np.min(all_ade, axis=0).mean()/self.pred_len
            fde_min = np.min(all_fde, axis=0).mean()
            ade_avg = np.mean(all_ade, axis=0).mean()/self.pred_len
            fde_avg = np.mean(all_fde, axis=0).mean()
            ade_std = np.std(all_ade, axis=0).mean()/self.pred_len
            fde_std = np.std(all_fde, axis=0).mean()

            sg_ade_min = np.min(sg_ade, axis=0).mean()/len(self.sg_idx)
            sg_ade_avg = np.mean(sg_ade, axis=0).mean()/len(self.sg_idx)
            sg_ade_std = np.std(sg_ade, axis=0).mean()/len(self.sg_idx)

            lg_fde_min = np.min(lg_fde, axis=0).mean()
            lg_fde_avg = np.mean(lg_fde, axis=0).mean()
            lg_fde_std = np.std(lg_fde, axis=0).mean()

        self.set_mode(train=True)
        return ade_min, fde_min, \
               ade_avg, fde_avg, \
               ade_std, fde_std, \
               sg_ade_min, sg_ade_avg, sg_ade_std, \
               lg_fde_min, lg_fde_avg, lg_fde_std



    def make_pred(self, data_loader, lg_num=5, traj_num=4, generate_heat=True):
        self.set_mode(train=False)
        total_traj = 0

        all_pred = []
        all_gt = []
        all_map = []
        all_homo = []
        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
                 obs_frames, pred_frames, map_path, inv_h_t,
                 local_map, local_ic, local_homo) = batch
                batch_size = obs_traj.size(1)
                total_traj += fut_traj.size(1)

                obs_heat_map, _, _= self.make_heatmap(local_ic, local_map)

                self.lg_cvae.forward(obs_heat_map, None, training=False)
                fut_rel_pos_dists = []
                pred_lg_wcs = []
                pred_sg_wcs = []

                ####### long term goals and the corresponding (deterministic) short term goals ########
                w_priors = []
                for _ in range(lg_num):
                    w_priors.append(self.lg_cvae.prior_latent_space.sample())

                for w_prior in w_priors:
                    # -------- long term goal --------
                    pred_lg_heat = F.sigmoid(self.lg_cvae.sample(self.lg_cvae.unet_enc_feat, w_prior))

                    pred_lg_wc = []
                    pred_lg_ics = []
                    for i in range(batch_size):
                        map_size = local_map[i][0].shape
                        pred_lg_ic = []
                        for heat_map in pred_lg_heat[i]:
                            argmax_idx = heat_map.argmax()
                            argmax_idx = [argmax_idx//map_size[0], argmax_idx%map_size[0]]
                            pred_lg_ic.append(argmax_idx)
                        pred_lg_ic = torch.tensor(pred_lg_ic).float().to(self.device)
                        pred_lg_ics.append(pred_lg_ic)

                        # ((local_ic[0,[11,15,19]] - pred_sg_ic) ** 2).sum(1).mean()
                        back_wc = torch.matmul(
                            torch.cat([pred_lg_ic, torch.ones((len(pred_lg_ic), 1)).to(self.device)], dim=1),
                            torch.transpose(local_homo[i].float().to(self.device), 1, 0))
                        pred_lg_wc.append(back_wc[0, :2] / back_wc[0, 2])
                        # ((back_wc - fut_traj[[3, 7, 11], 0, :2]) ** 2).sum(1).mean()
                    pred_lg_wc = torch.stack(pred_lg_wc)
                    pred_lg_wcs.append(pred_lg_wc)

                    # -------- short term goal --------
                    # obs_lg_heat = torch.cat([obs_heat_map, pred_lg_heat[:, -1].unsqueeze(1)], dim=1)

                    if generate_heat:
                        # -------- short term goal --------
                        pred_lg_heat_from_ic = []
                        for coord in pred_lg_ics:
                            heat_map_traj = np.zeros((160, 160))
                            heat_map_traj[int(coord[0, 0]), int(coord[0, 1])] = 1
                            heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
                            pred_lg_heat_from_ic.append(heat_map_traj)
                        pred_lg_heat_from_ic = torch.tensor(np.stack(pred_lg_heat_from_ic)).unsqueeze(1).float().to(
                            self.device)

                        pred_sg_heat = F.sigmoid(self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat_from_ic], dim=1)))
                    else:
                        pred_sg_heat = F.sigmoid(self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat], dim=1)))


                    pred_sg_wc = []
                    for i in range(batch_size):
                        pred_sg_ic = []
                        for heat_map in pred_sg_heat[i]:
                            argmax_idx = heat_map.argmax()
                            argmax_idx = [argmax_idx//map_size[0], argmax_idx%map_size[0]]
                            pred_sg_ic.append(argmax_idx)
                        pred_sg_ic = torch.tensor(pred_sg_ic).float().to(self.device)

                        # ((local_ic[0,[11,15,19]] - pred_sg_ic) ** 2).sum(1).mean()
                        back_wc = torch.matmul(
                            torch.cat([pred_sg_ic, torch.ones((len(pred_sg_ic), 1)).to(self.device)], dim=1),
                            torch.transpose(local_homo[i].float().to(self.device), 1, 0))
                        back_wc /= back_wc[:, 2].unsqueeze(1)
                        pred_sg_wc.append(back_wc[:, :2])
                        # ((back_wc - fut_traj[[3, 7, 11], 0, :2]) ** 2).sum(1).mean()
                    pred_sg_wc = torch.stack(pred_sg_wc)
                    pred_sg_wcs.append(pred_sg_wc)

                ##### trajectories per long&short goal ####

                # -------- trajectories --------
                (hx, mux, log_varx) \
                    = self.encoderMx(obs_traj_st, seq_start_end, self.lg_cvae.unet_enc_feat, local_homo)

                p_dist = Normal(mux, torch.sqrt(torch.exp(log_varx)))
                z_priors = []
                for _ in range(traj_num):
                    z_priors.append(p_dist.sample())

                for pred_sg_wc in pred_sg_wcs:
                    for z_prior in z_priors:
                        # -------- trajectories --------
                        # NO TF, pred_goals, z~prior
                        fut_rel_pos_dist_prior = self.decoderMy(
                            obs_traj_st[-1],
                            obs_traj[-1, :, :2],
                            hx,
                            z_prior,
                            pred_sg_wc,  # goal
                            self.sg_idx
                        )
                        fut_rel_pos_dists.append(fut_rel_pos_dist_prior)
                pred = []
                for dist in fut_rel_pos_dists:
                    pred_fut_traj = integrate_samples(dist.rsample(), obs_traj[-1, :, :2],  dt=self.dt)
                    pred.append(pred_fut_traj)

                all_pred.append(np.stack(pred).transpose(2,0,1,3))
                all_gt.append(fut_traj[:, :, :2].transpose(1,0).detach().cpu().numpy())
                all_map.append(1-local_map.squeeze(1))
                all_homo.append(local_homo)

        import pickle
        # data = [np.concatenate(all_pred, -2).transpose(0, 2, 1, 3),
        #         np.concatenate(all_gt, -2).transpose(0, 2, 1, 3)]
        # with open('path_' + str(traj_num * lg_num) + '.pkl', 'wb') as handle:
        #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        data = [np.concatenate(all_pred, 0),
                np.concatenate(all_gt, 0), np.concatenate(all_map, 0), np.concatenate(all_homo, 0)]
        with open('path_c_' + str(traj_num * lg_num) + '.pkl', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)



    def make_pred_lg(self, data_loader, lg_num=5, traj_num=4, generate_heat=True):
        self.set_mode(train=False)
        total_traj = 0
        all_pred = []
        all_gt = []
        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
                 obs_frames, pred_frames, map_path, inv_h_t,
                 local_map, local_ic, local_homo) = batch
                batch_size = obs_traj.size(1)
                total_traj += fut_traj.size(1)

                obs_heat_map, _, _= self.make_heatmap(local_ic, local_map)

                self.lg_cvae.forward(obs_heat_map, None, training=False)
                fut_rel_pos_dists = []
                pred_lg_wcs = []
                pred_sg_wcs = []

                ####### long term goals and the corresponding (deterministic) short term goals ########
                w_priors = []
                for _ in range(lg_num):
                    w_priors.append(self.lg_cvae.prior_latent_space.sample())

                for w_prior in w_priors:
                    # -------- long term goal --------
                    pred_lg_heat = F.sigmoid(self.lg_cvae.sample(self.lg_cvae.unet_enc_feat, w_prior))

                    pred_lg_wc = []
                    pred_lg_ics = []
                    for i in range(batch_size):
                        map_size = local_map[i][0].shape
                        pred_lg_ic = []
                        for heat_map in pred_lg_heat[i]:
                            argmax_idx = heat_map.argmax()
                            argmax_idx = [argmax_idx//map_size[0], argmax_idx%map_size[0]]
                            pred_lg_ic.append(argmax_idx)
                        pred_lg_ic = torch.tensor(pred_lg_ic).float().to(self.device)
                        pred_lg_ics.append(pred_lg_ic)

                        # ((local_ic[0,[11,15,19]] - pred_sg_ic) ** 2).sum(1).mean()
                        back_wc = torch.matmul(
                            torch.cat([pred_lg_ic, torch.ones((len(pred_lg_ic), 1)).to(self.device)], dim=1),
                            torch.transpose(local_homo[i].float().to(self.device), 1, 0))
                        pred_lg_wc.append(back_wc[0, :2] / back_wc[0, 2])
                        # ((back_wc - fut_traj[[3, 7, 11], 0, :2]) ** 2).sum(1).mean()
                    pred_lg_wc = torch.stack(pred_lg_wc)
                    pred_lg_wcs.append(pred_lg_wc)

                # -------- trajectories --------
                (hx, mux, log_varx) \
                    = self.encoderMx(obs_traj_st, seq_start_end, self.lg_cvae.unet_enc_feat, local_homo)

                p_dist = Normal(mux, torch.sqrt(torch.exp(log_varx)))
                z_priors = []
                for _ in range(traj_num):
                    z_priors.append(p_dist.sample())

                for pred_lg_wc in pred_lg_wcs:
                    for z_prior in z_priors:
                        # -------- trajectories --------
                        # NO TF, pred_goals, z~prior
                        fut_rel_pos_dist_prior = self.decoderMy(
                            obs_traj_st[-1],
                            obs_traj[-1, :, :2],
                            hx,
                            z_prior,
                            pred_lg_wc.unsqueeze(1),  # goal
                            self.sg_idx
                        )
                        fut_rel_pos_dists.append(fut_rel_pos_dist_prior)
                pred = []
                for dist in fut_rel_pos_dists:
                    pred_fut_traj = integrate_samples(dist.rsample(), obs_traj[-1, :, :2],  dt=self.dt)
                    pred.append(pred_fut_traj)
                all_pred.append(torch.stack(pred).detach().cpu().numpy())
                all_gt.append(
                    fut_traj[:, :, :2].unsqueeze(0).repeat((traj_num * lg_num, 1, 1, 1)).detach().cpu().numpy())

        import pickle
        data = [np.concatenate(all_pred, -2).transpose(0, 2, 1, 3),
                np.concatenate(all_gt, -2).transpose(0, 2, 1, 3)]
        with open('lg_path_' + str(traj_num * lg_num) + '.pkl', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open('D:\crowd\datasets\Trajectories\/test.pickle', 'rb') as f:
        #     a = pickle.load(f)



    def make_pred_12sg(self, data_loader, lg_num=5, traj_num=4, generate_heat=True):
        self.set_mode(train=False)
        total_traj = 0

        all_pred = []
        all_gt = []
        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
                 obs_frames, pred_frames, map_path, inv_h_t,
                 local_map, local_ic, local_homo) = batch
                batch_size = obs_traj.size(1)
                total_traj += fut_traj.size(1)

                obs_heat_map, _, _= self.make_heatmap(local_ic, local_map)

                self.lg_cvae.forward(obs_heat_map, None, training=False)
                fut_rel_pos_dists = []
                pred_lg_wcs = []
                pred_sg_wcs = []

                ####### long term goals and the corresponding (deterministic) short term goals ########
                w_priors = []
                for _ in range(lg_num):
                    w_priors.append(self.lg_cvae.prior_latent_space.sample())

                for w_prior in w_priors:
                    # -------- long term goal --------
                    pred_lg_heat = F.sigmoid(self.lg_cvae.sample(self.lg_cvae.unet_enc_feat, w_prior))

                    pred_lg_wc = []
                    pred_lg_ics = []
                    for i in range(batch_size):
                        map_size = local_map[i][0].shape
                        pred_lg_ic = []
                        for heat_map in pred_lg_heat[i]:
                            argmax_idx = heat_map.argmax()
                            argmax_idx = [argmax_idx//map_size[0], argmax_idx%map_size[0]]
                            pred_lg_ic.append(argmax_idx)
                        pred_lg_ic = torch.tensor(pred_lg_ic).float().to(self.device)
                        pred_lg_ics.append(pred_lg_ic)

                        # ((local_ic[0,[11,15,19]] - pred_sg_ic) ** 2).sum(1).mean()
                        back_wc = torch.matmul(
                            torch.cat([pred_lg_ic, torch.ones((len(pred_lg_ic), 1)).to(self.device)], dim=1),
                            torch.transpose(local_homo[i].float().to(self.device), 1, 0))
                        pred_lg_wc.append(back_wc[0, :2] / back_wc[0, 2])
                        # ((back_wc - fut_traj[[3, 7, 11], 0, :2]) ** 2).sum(1).mean()
                    pred_lg_wc = torch.stack(pred_lg_wc)
                    pred_lg_wcs.append(pred_lg_wc)

                    # -------- short term goal --------
                    # obs_lg_heat = torch.cat([obs_heat_map, pred_lg_heat[:, -1].unsqueeze(1)], dim=1)

                    if generate_heat:
                        # -------- short term goal --------
                        pred_lg_heat_from_ic = []
                        for coord in pred_lg_ics:
                            heat_map_traj = np.zeros((160, 160))
                            heat_map_traj[int(coord[0, 0]), int(coord[0, 1])] = 1
                            heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
                            pred_lg_heat_from_ic.append(heat_map_traj)
                        pred_lg_heat_from_ic = torch.tensor(np.stack(pred_lg_heat_from_ic)).unsqueeze(1).float().to(
                            self.device)

                        pred_sg_heat = F.sigmoid(self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat_from_ic], dim=1)))
                    else:
                        pred_sg_heat = F.sigmoid(self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat], dim=1)))


                    pred_sg_wc = []
                    for i in range(batch_size):
                        pred_sg_ic = []
                        for heat_map in pred_sg_heat[i]:
                            argmax_idx = heat_map.argmax()
                            argmax_idx = [argmax_idx//map_size[0], argmax_idx%map_size[0]]
                            pred_sg_ic.append(argmax_idx)
                        pred_sg_ic = torch.tensor(pred_sg_ic).float().to(self.device)

                        # ((local_ic[0,[11,15,19]] - pred_sg_ic) ** 2).sum(1).mean()
                        back_wc = torch.matmul(
                            torch.cat([pred_sg_ic, torch.ones((len(pred_sg_ic), 1)).to(self.device)], dim=1),
                            torch.transpose(local_homo[i].float().to(self.device), 1, 0))
                        back_wc /= back_wc[:, 2].unsqueeze(1)
                        pred_sg_wc.append(back_wc[:, :2])
                        # ((back_wc - fut_traj[[3, 7, 11], 0, :2]) ** 2).sum(1).mean()
                    pred_sg_wc = torch.stack(pred_sg_wc)
                    pred_sg_wcs.append(pred_sg_wc)

                all_pred.append(torch.stack(pred_sg_wcs).permute(0, 2, 1, 3).detach().cpu().numpy())
                all_gt.append(
                    fut_traj[:, :, :2].unsqueeze(0).repeat((traj_num * lg_num, 1, 1, 1)).detach().cpu().numpy())

        import pickle
        print('pred:', np.concatenate(all_pred, -2).transpose(0, 2, 1, 3).shape)
        print('gt:', np.concatenate(all_gt, -2).transpose(0, 2, 1, 3).shape)
        data = [np.concatenate(all_pred, -2).transpose(0, 2, 1, 3),
                np.concatenate(all_gt, -2).transpose(0, 2, 1, 3)]
        with open('12sg_path_' + str(traj_num * lg_num) + '.pkl', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open('D:\crowd\datasets\Trajectories\/test.pickle', 'rb') as f:
        #     a = pickle.load(f)




    def set_mode(self, train=True):

        if train:
            self.sg_unet.train()
            self.lg_cvae.train()
            self.encoderMx.train()
            self.encoderMy.train()
            self.decoderMy.train()
        else:
            self.sg_unet.eval()
            self.lg_cvae.eval()
            self.encoderMx.eval()
            self.encoderMy.eval()
            self.decoderMy.eval()

    ####
    def save_checkpoint(self, iteration):

        sg_unet_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_sg_unet.pt' % iteration
        )
        mkdirs(self.ckpt_dir)
        torch.save(self.sg_unet, sg_unet_path)



    def pretrain_load_checkpoint(self, traj, lg, sg):
        sg_unet_path = os.path.join(
            sg['ckpt_dir'],
            'iter_%s_sg_unet.pt' % sg['iter']
        )


        encoderMx_path = os.path.join(
            traj['ckpt_dir'],
            'iter_%s_encoderMx.pt' % traj['iter']
        )
        encoderMy_path = os.path.join(
            traj['ckpt_dir'],
            'iter_%s_encoderMy.pt' % traj['iter']
        )
        decoderMy_path = os.path.join(
            traj['ckpt_dir'],
            'iter_%s_decoderMy.pt' %  traj['iter']
        )
        lg_cvae_path = os.path.join(
            lg['ckpt_dir'],
            'iter_%s_lg_cvae.pt' %  lg['iter']
        )


        if self.device == 'cuda':
            self.encoderMx = torch.load(encoderMx_path)
            self.encoderMy = torch.load(encoderMy_path)
            self.decoderMy = torch.load(decoderMy_path)
            self.lg_cvae = torch.load(lg_cvae_path)
            self.sg_unet = torch.load(sg_unet_path)

        else:
            self.encoderMx = torch.load(encoderMx_path, map_location='cpu')
            self.encoderMy = torch.load(encoderMy_path, map_location='cpu')
            self.decoderMy = torch.load(decoderMy_path, map_location='cpu')
            self.lg_cvae = torch.load(lg_cvae_path, map_location='cpu')
            self.sg_unet = torch.load(sg_unet_path, map_location='cpu')