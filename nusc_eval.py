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
import cv2
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from data.nuscenes.config import Config
from data.nuscenes_dataloader import data_generator
import numpy as np
import visdom
import torch.nn.functional as F
import torch.nn.functional as nnf


###############################################################################

def compute_ECFL(output_traj, binary_navmaps):
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
            for t in range(output_traj.shape[2]):
                pos = output_traj[i, k, t]
                if pos[1] < 0 or pos[1] >= binary_navmaps[i].shape[1] or pos[0] < 0 or pos[0] >= binary_navmaps[i].shape[0]:
                    collided = True
                    break
                if binary_navmaps[i][pos[0], pos[1]] == 0:
                    collided = True
                    break

            if not collided:
                ecfl += 1.0 / output_traj.shape[1]

    return ecfl / output_traj.shape[0]


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


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class Solver(object):

    ####
    def __init__(self, args):

        self.args = args
        args.num_sg = args.load_e
        self.name = '%s_enc_block_%s_fcomb_block_%s_wD_%s_lr_%s_lg_klw_%s_a_%s_r_%s_fb_%s_anneal_e_%s_load_e_%s_aug_%s_scael_%s' % \
                    (args.dataset_name, args.no_convs_per_block, args.no_convs_fcomb, args.w_dim, args.lr_VAE,
                     args.lg_kl_weight, args.alpha, args.gamma, args.fb, args.anneal_epoch, args.load_e, args.aug, args.scale)
        # self.name = 'sg_enc_block_1_fcomb_block_2_wD_10_lr_0.001_lg_klw_1_a_0.25_r_2.0_fb_2.0_anneal_e_10_load_e_1'

        # to be appended by run_id

        # self.use_cuda = args.cuda and torch.cuda.is_available()
        self.fb = args.fb
        self.scale = args.scale
        self.anneal_epoch = args.anneal_epoch
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.aug = args.aug
        self.device = args.device
        self.temp=1.99
        self.dt=0.5
        self.eps=1e-9
        self.ll_prior_w =args.ll_prior_w
        self.sg_idx = np.array(range(12))
        self.sg_idx = np.flip(11-self.sg_idx[::(12//args.num_sg)])
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


        #### create a new model or load a previously saved model

        self.ckpt_load_iter = args.ckpt_load_iter

        self.obs_len = 4
        self.pred_len = args.pred_len
        self.num_layers = args.num_layers
        self.decoder_h_dim = args.decoder_h_dim

        if self.ckpt_load_iter != self.max_iter:
            cfg = Config('nuscenes_train', False, create_dirs=True)
            torch.set_default_dtype(torch.float32)
            log = open('log.txt', 'a+')
            self.train_loader = data_generator(cfg, log, split='train', phase='training',
                                               batch_size=args.batch_size, device=self.device, scale=args.scale, shuffle=True)

            cfg = Config('nuscenes', False, create_dirs=True)
            torch.set_default_dtype(torch.float32)
            log = open('log.txt', 'a+')
            self.val_loader = data_generator(cfg, log, split='test', phase='testing',
                                             batch_size=args.batch_size, device=self.device, scale=args.scale, shuffle=True)

            print(
                'There are {} iterations per epoch'.format(len(self.train_loader.idx_list))
            )
        print('...done')




    def make_heatmap(self, local_ic, local_map, aug=False):
        heat_maps=[]
        down_size=256
        half = down_size//2
        for i in range(len(local_ic)):
            '''
            plt.imshow(local_map[i])
            plt.scatter(local_ic[i,:4,1], local_ic[i,:4,0], s=1, c='b')
            plt.scatter(local_ic[i,4:,1], local_ic[i,4:,0], s=1, c='g')
            '''
            map_size = local_map[i].shape[0]
            if map_size < down_size:
                env = np.full((down_size,down_size),1)
                env[half-map_size//2:half+map_size//2, half-map_size//2:half+map_size//2] = local_map[i]
                ohm = [env]
                heat_map_traj = np.zeros_like(local_map[i])
                heat_map_traj[local_ic[i, :self.obs_len, 0], local_ic[i, :self.obs_len, 1]] = 1
                heat_map_traj= ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
                heat_map_traj = heat_map_traj / heat_map_traj.sum()
                extended_map = np.zeros((down_size, down_size))
                extended_map[half-map_size//2:half+map_size//2, half-map_size//2:half+map_size//2] = heat_map_traj
                ohm.append(extended_map)
                # future
                for j in (self.sg_idx + self.obs_len):
                    heat_map_traj = np.zeros_like(local_map[i])
                    heat_map_traj[local_ic[i, j, 0], local_ic[i, j, 1]] = 1
                    heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
                    extended_map = np.zeros((down_size, down_size))
                    extended_map[half-map_size//2:half+map_size//2, half-map_size//2:half+map_size//2]= heat_map_traj
                    ohm.append(extended_map)
                heat_maps.append(np.stack(ohm))
            else:
                env = cv2.resize(local_map[i], dsize=(down_size, down_size))
                ohm = [env]
                heat_map_traj = np.zeros_like(local_map[i])
                heat_map_traj[local_ic[i, :self.obs_len, 0], local_ic[i, :self.obs_len, 1]] = 100

                if map_size > 1000:
                    heat_map_traj = cv2.resize(ndimage.filters.gaussian_filter(heat_map_traj, sigma=2),
                                               dsize=((map_size+down_size)//2, (map_size+down_size)//2))
                    heat_map_traj = heat_map_traj / heat_map_traj.sum()
                heat_map_traj = cv2.resize(ndimage.filters.gaussian_filter(heat_map_traj, sigma=2), dsize=(down_size, down_size))
                if map_size > 3500:
                    heat_map_traj[np.where(heat_map_traj > 0)] = 1
                else:
                    heat_map_traj = heat_map_traj / heat_map_traj.sum()
                heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
                ohm.append(heat_map_traj / heat_map_traj.sum())

                '''
                heat_map = nnf.interpolate(torch.tensor(heat_map_traj).unsqueeze(0).unsqueeze(0),
                                           size=local_map[i].shape, mode='nearest').squeeze(0).squeeze(0)
                heat_map = nnf.interpolate(torch.tensor(heat_map_traj).unsqueeze(0).unsqueeze(0),
                                           size=local_map[i].shape,  mode='bicubic',
                                                  align_corners = False).squeeze(0).squeeze(0)
                '''
                for j in (self.sg_idx+ self.obs_len):
                    heat_map_traj = np.zeros_like(local_map[i])
                    heat_map_traj[local_ic[i, j, 0], local_ic[i, j, 1]] = 1000
                    if map_size > 1000:
                        heat_map_traj = cv2.resize(ndimage.filters.gaussian_filter(heat_map_traj, sigma=2),
                                                   dsize=((map_size+down_size)//2, (map_size+down_size)//2))
                    heat_map_traj = cv2.resize(ndimage.filters.gaussian_filter(heat_map_traj, sigma=2), dsize=(down_size, down_size))
                    heat_map_traj = heat_map_traj / heat_map_traj.sum()
                    heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
                    ohm.append(heat_map_traj)
                heat_maps.append(np.stack(ohm))

        heat_maps = torch.tensor(np.stack(heat_maps)).float().to(self.device)

        if aug:
            degree = np.random.choice([0,90,180, -90])
            heat_maps = transforms.Compose([
                transforms.RandomRotation(degrees=(degree, degree))
            ])(heat_maps)
        return heat_maps[:,:2], heat_maps[:,2:], heat_maps[:,-1].unsqueeze(1)

    def make_one_heatmap(self, local_map, local_ic):
        map_size = local_map.shape[0]
        down_size=256
        half = down_size // 2
        if map_size < down_size:
            heat_map_traj = np.zeros_like(local_map)
            heat_map_traj[local_ic[0], local_ic[1]] = 1
            heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
            extended_map = np.zeros((down_size, down_size))
            extended_map[half - map_size // 2:half + map_size // 2,
            half - map_size // 2:half + map_size // 2] = heat_map_traj
        else:
            heat_map_traj = np.zeros_like(local_map)
            heat_map_traj[local_ic[0], local_ic[1]] = 1000
            if map_size > 1000:
                heat_map_traj = cv2.resize(ndimage.filters.gaussian_filter(heat_map_traj, sigma=2),
                                           dsize=((map_size + down_size) // 2, (map_size + down_size) // 2))
            heat_map_traj = cv2.resize(ndimage.filters.gaussian_filter(heat_map_traj, sigma=2),
                                       dsize=(down_size, down_size))
            heat_map_traj = heat_map_traj / heat_map_traj.sum()
            heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
        return heat_map_traj




    ####
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
            while not data_loader.is_epoch_end():
                data = data_loader.next_sample()
                if data is None:
                    continue
                b+=1

                idx = 412
                data_loader.index = idx
                data = data_loader.next_sample()

                (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
                 maps, local_map, local_ic, local_homo) = data
                plt.imshow(local_map[0])
                plt.scatter(local_ic[0,:,1], local_ic[0,:,0], s=1, c='r')
                plt.show()
                #
                # batch = data_loader.dataset.__getitem__(143)
                # (obs_traj, fut_traj,
                #  obs_frames, pred_frames, map_path, inv_h_t,
                #  local_map, local_ic, local_homo) = batch

                obs_heat_map, sg_heat_map, lg_heat_map = self.make_heatmap(local_ic, local_map)

                self.lg_cvae.forward(obs_heat_map, None, training=False)

                ###################################################
                i = 0

                # ori_map = maps[0].data.transpose(1,2,0)
                # plt.imshow(ori_map)
                # fig, ax = maps[0].render_map_patch(my_patch, layers, figsize=(10, 10), alpha=0.1,
                #                                     render_egoposes_range=False)
                #
                # fig, ax = maps[0].render_map_patch([0,0,ori_map.shape[0],ori_map.shape[1]], layers=None, figsize=(10, 10), alpha=0.1,
                #                                     render_egoposes_range=False)
                #
                # plt.imshow()

                ####################### LG ############################

                zs = []
                for _ in range(10):
                    zs.append(self.lg_cvae.prior_latent_space.rsample())

                mm = []
                for k in range(5):
                    mm.append(F.sigmoid(self.lg_cvae.sample(self.lg_cvae.unet_enc_feat, zs[k])))
                # mm.append(F.sigmoid(self.lg_cvae.sample(self.lg_cvae.unet_enc_feat, self.lg_cvae.posterior_latent_space.rsample())))

                mmm = []
                for k in range(5, 10):
                    mmm.append(F.sigmoid(self.lg_cvae.sample(self.lg_cvae.unet_enc_feat, zs[k])))
                # mm.append(F.sigmoid(self.lg_cvae.sample(self.lg_cvae.unet_enc_feat, self.lg_cvae.posterior_latent_space.rsample())))

                # ------- plot -----------
                env = local_map[i]
                # env = cv2.resize(env, (256,256))
                # for t in [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 19]:
                #     env[local_ic[i, t, 0], local_ic[i, t, 1]] = 0

                heat_map_traj = np.zeros_like(env)
                for t in [0, 1, 2, 3, 7, 11, 15]:
                    heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 50
                heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=1)
                heat_map_traj = 1 - heat_map_traj * 3 / heat_map_traj.max()

                # plt.imshow(np.stack([heat_map_traj, heat_map_traj, heat_map_traj], axis=2))

                all_pred = []
                for k in range(5):
                    a = mm[k][i, 0]
                    a = nnf.interpolate(torch.tensor(a).unsqueeze(0).unsqueeze(0),
                                        size=local_map[i].shape, mode='bicubic',
                                        align_corners=False).squeeze(0).squeeze(0).detach().cpu().numpy().copy()
                    all_pred.append(1 - a / a.max())
                for k in range(5):
                    a = mmm[k][i, 0].detach().cpu().numpy().copy()
                    a = nnf.interpolate(torch.tensor(a).unsqueeze(0).unsqueeze(0),
                                        size=local_map[i].shape, mode='bicubic',
                                        align_corners=False).squeeze(0).squeeze(0).detach().cpu().numpy().copy()
                    all_pred.append(1 - a / a.max())

                fig = plt.figure(figsize=(12, 10))
                fig.tight_layout()
                for k in range(10):
                    ax = fig.add_subplot(4, 5, k + 1)
                    ax.set_title('prior' + str(k % 5 + 1))
                    if k < 5:
                        # ax.imshow(np.stack([env * (1 - heat_map_traj), env * (1 - a * 5), env], axis=2))
                        # ax.imshow(np.stack([(1 - heat_map_traj*1000), (1 - all_pred[k]*1000), env/env.max()], axis=2))
                        ax.imshow(np.stack([heat_map_traj, all_pred[k], 1 - env], axis=2))
                    else:
                        ax.imshow(mm[k % 5][i, 0])

                for k in range(10):
                    ax = fig.add_subplot(4, 5, k + 11)
                    ax.set_title('prior' + str(k % 5 + 6))
                    if k < 5:
                        ax.imshow(np.stack([heat_map_traj, all_pred[k + 5], 1 - env], axis=2))
                    else:
                        ax.imshow(mmm[k % 5][i, 0])

                plt.imshow(env)


                tmp_idx = i
                ####################### SG ############################

                pred_lg_heat = mm[3]
                pred_lg_ics = []
                pred_lg_wc = []
                for i in range(len(obs_heat_map)):
                    map_size = local_map[i].shape
                    pred_lg_ic = []
                    for heat_map in pred_lg_heat[i]:
                        # heat_map = nnf.interpolate(heat_map.unsqueeze(0), size=map_size, mode='nearest')
                        heat_map = nnf.interpolate(heat_map.unsqueeze(0).unsqueeze(0),
                                                   size=map_size, mode='bicubic',
                                                   align_corners=False).squeeze(0).squeeze(0)
                        argmax_idx = heat_map.argmax()
                        argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                        pred_lg_ic.append(argmax_idx)

                    pred_lg_ic = torch.tensor(pred_lg_ic).float().to(self.device)

                    pred_lg_ics.append(pred_lg_ic)

                    # ((local_ic[0,[11,15,19]] - pred_sg_ic) ** 2).sum(1).mean()
                    back_wc = torch.matmul(
                        torch.cat([pred_lg_ic, torch.ones((len(pred_lg_ic), 1)).to(self.device)], dim=1),
                        torch.transpose(local_homo[i], 1, 0))
                    pred_lg_wc.append(back_wc[0, :2] / back_wc[0, 2])
                    # ((back_wc - fut_traj[[3, 7, 11], 0, :2]) ** 2).sum(1).mean()
                pred_lg_wc = torch.stack(pred_lg_wc)

                # -------- short term goal --------
                pred_lg_heat_from_ic = []
                for lg_idx in range(len(pred_lg_ics)):
                    pred_lg_heat_from_ic.append(self.make_one_heatmap(local_map[lg_idx], pred_lg_ics[lg_idx].detach().cpu().numpy().astype(int)[0]))
                pred_lg_heat_from_ic = torch.tensor(np.stack(pred_lg_heat_from_ic)).unsqueeze(1).float().to(
                    self.device)
                pred_sg = F.sigmoid(
                    self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat_from_ic], dim=1)))

                i =tmp_idx
                m = pred_sg.detach().cpu().numpy().copy()[i]
                m = nnf.interpolate(torch.tensor(m).unsqueeze(0),
                                    size=local_map[i].shape, mode='bicubic',
                                    align_corners=False).squeeze(0).detach().cpu().numpy().copy()

                fig = plt.figure(figsize=(10, 5))

                for k in range(6):
                    ax = fig.add_subplot(2,3,k+1)
                    ax.set_title('sg' + str(k+1))
                    if k <3:
                        # ax.imshow(np.stack([env * (1 - heat_map_traj), env * (1 - m * 5), env], axis=2))
                        ax.imshow(np.stack([heat_map_traj, 1-m[k] / m[k].max(), 1 - env], axis=2))
                        # ax.imshow(np.stack([heat_map_traj,1 - a / a.max(), 1 - env], axis=2))


                    else:
                        ax.imshow(m[k%3])

                #################### GIF #################### GIF

                pred_lg_wcs = []
                pred_sg_wcs = []
                traj_num=1
                lg_num=10
                pred_lg_heats= []
                pred_sg_heats= []
                for j in range(lg_num):
                    # set_seed(j)
                    # -------- long term goal --------
                    pred_lg_heat = F.sigmoid(self.lg_cvae.sample(testing=True))
                    pred_lg_heats.append(pred_lg_heat)
                    pred_lg_ics = []
                    pred_lg_wc = []
                    for i in range(len(obs_heat_map)):
                        map_size = local_map[i].shape
                        pred_lg_ic = []
                        for heat_map in pred_lg_heat[i]:
                            # heat_map = nnf.interpolate(heat_map.unsqueeze(0), size=map_size, mode='nearest')
                            heat_map = nnf.interpolate(heat_map.unsqueeze(0).unsqueeze(0),
                                                       size=map_size, mode='bicubic',
                                                       align_corners=False).squeeze(0).squeeze(0)
                            argmax_idx = heat_map.argmax()
                            argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                            pred_lg_ic.append(argmax_idx)

                        pred_lg_ic = torch.tensor(pred_lg_ic).float().to(self.device)

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
                    for i in range(len(pred_lg_ics)):
                        pred_lg_heat_from_ic.append(self.make_one_heatmap(local_map[i], pred_lg_ics[i][
                            0].detach().cpu().numpy().astype(int)))
                    pred_lg_heat_from_ic = torch.tensor(np.stack(pred_lg_heat_from_ic)).unsqueeze(1).float().to(
                        self.device)
                    pred_sg_heat = F.sigmoid(
                        self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat_from_ic], dim=1)))
                    pred_sg_heats.append(pred_sg_heat)


                    pred_sg_wc = []
                    for i in range(len(obs_heat_map)):
                        map_size = local_map[i].shape
                        pred_sg_ic = []
                        for heat_map in pred_sg_heat[i]:
                            heat_map = nnf.interpolate(heat_map.unsqueeze(0).unsqueeze(0),
                                                       size=map_size, mode='bicubic',
                                                       align_corners=False).squeeze(0).squeeze(0)
                            argmax_idx = heat_map.argmax()
                            argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                            pred_sg_ic.append(argmax_idx)
                        pred_sg_ic = torch.tensor(pred_sg_ic).float().to(self.device)
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
                i = tmp_idx

                multi_sample_pred = []

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

                    one_ped = tmp_idx

                    pred_real = pred_fut_traj[:, one_ped].numpy()
                    pred_pixel = np.concatenate([pred_real, np.ones((self.pred_len, 1))],
                                                axis=1)

                    pred_pixel = np.matmul(pred_pixel, np.linalg.inv(np.transpose(local_homo[one_ped])))
                    pred_pixel /= np.expand_dims(pred_pixel[:, 2], 1)
                    pred_data.append(np.concatenate([local_ic[i,:4], pred_pixel[:,:2]], 0))

                pred_data = np.expand_dims(np.stack(pred_data),1) #  (20, 1, 16, 2)

                #---------- plot gif
                '''
                i = tmp_idx

                env = local_map[i]
                # env = np.stack([env, env, env], axis=2)

                def init():
                    ax.imshow(env)

                def update_dot(num_t):
                    print(num_t)
                    ax.imshow(env)
                    for j in range(len(pred_data)):
                        ln_pred[j].set_data(pred_data[j, 0, :num_t, 1],
                                                   pred_data[j, 0, :num_t, 0])
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



                ani = FuncAnimation(fig, update_dot, frames=16, interval=1, init_func=init())

                # writer = PillowWriter(fps=3000)
                gif_path = 'D:\crowd\datasets\Trajectories'
                ani.save(gif_path + "/" "path_find_agent" + str(i) + ".gif", fps=4)
                '''
                #====================================================================================================
                #==================================================for report ==================================================
                #====================================================================================================


                ######################


                import matplotlib.patheffects as pe
                import seaborn as sns
                from matplotlib.offsetbox import OffsetImage, AnnotationBbox
                ####### plot 20 trajs ############
                i = tmp_idx
                env = np.expand_dims((1-local_map[i]), 2).repeat(3,2)
                # env = 1-local_map[i]

                # env= np.eye(256,256)
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

                for t in range(self.obs_len,self.obs_len+self.pred_len):
                    # sns.kdeplot(pred_data[:, 0, t, 1], pred_data[:, 0, t, 0],shade=True)
                    sns.kdeplot(pred_data[:, 0, t, 1], pred_data[:, 0, t, 0],
                                ax=ax, shade=True, shade_lowest=False, bw=None,
                                color='r', zorder=600, alpha=0.3, legend=True)

                # ax.scatter(local_ic[0,:,1], local_ic[0,:,0], s=5, c='b')

                for t in range(10):
                    if t ==0:
                        ax.plot(pred_data[t, 0, 4:, 1], pred_data[t, 0, 4:, 0], 'r--', linewidth=2,
                                zorder=650, c='firebrick',
                                path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()], label='Predicted future')
                    else:
                        ax.plot(pred_data[t, 0, 4:, 1], pred_data[t, 0, 4:, 0], 'r--', linewidth=2,
                                zorder=650, c='firebrick',
                                path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])

                ax.plot(local_ic[0,4:,1], local_ic[0,4:,0],
                        'r--o',
                        c='darkorange',
                        linewidth=2,
                        markersize=2,
                        zorder=650,
                        path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()], label='GT future')
                ax.plot(local_ic[0,:4,1], local_ic[0,:4,0],
                        'b--o',
                        c='royalblue',
                        linewidth=2,
                        markersize=2,
                        zorder=650,
                        path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()], label='GT past')

                ax.legend(loc='upper left')


                ################### LG ##############################
                # ------- plot -----------
                env = np.expand_dims((1 - local_map[i]), 2).repeat(3, 2)

                fig = plt.figure(figsize=(12, 10))
                idx_list = [2,3,5]
                # idx_list = [7,8,9]
                # idx_list = [4,5,6]
                for h in range(3):
                    idx = idx_list[h]
                    ax = fig.add_subplot(3, 4, 4*h+1)
                    plt.tight_layout()
                    ax.imshow(env)
                    ax.axis('off')
                    # ax.scatter(local_ic[0,:4,1], local_ic[0,:4,0], c='b', s=2, alpha=0.7)
                    # ax.scatter(local_ic[0,4:,1], local_ic[0,4:,0], c='r', s=2, alpha=0.7)
                    # ax.scatter(local_ic[0,-1,1], local_ic[0,-1,0], c='r', s=18, marker='x', alpha=0.7)

                    ax.plot(local_ic[0, :4, 1], local_ic[0, :4, 0],
                            'b--o',
                            c='royalblue',
                            linewidth=1,
                            markersize=1.5,
                            zorder=500,
                            path_effects=[pe.Stroke(linewidth=1, foreground='k'), pe.Normal()], label='GT past')
                    ax.plot(local_ic[0, 4:, 1], local_ic[0, 4:, 0],
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
                            markersize=5,
                            zorder=500,
                            path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()], label='GT past')

                    a = pred_lg_heats[idx][i, 0]
                    a = nnf.interpolate(torch.tensor(a).unsqueeze(0).unsqueeze(0),
                                        size=local_map[i].shape, mode='bicubic',
                                        align_corners=False).squeeze(0).squeeze(0).detach().cpu().numpy().copy()

                    # b = a / a.max()
                    # b = np.expand_dims((1 - b), 2).repeat(3, 2)
                    c = a / a.max()
                    # env = np.expand_dims((1 - local_map[i]), 2).repeat(3, 2)
                    # env[:,:,0][np.where(c>0.1)] = 1-c[np.where(c>0.1)]
                    # plt.imshow(env)
                    d=np.stack([1-c, np.ones_like(c), np.ones_like(c)]).transpose(1,2,0)
                    ax.imshow(d, alpha=0.7)

                    # -------- short term goal --------

                    pred_lg_heat = pred_lg_heats[idx]
                    pred_lg_ics = []
                    for j in range(len(obs_heat_map)):
                        map_size = local_map[j].shape
                        pred_lg_ic = []
                        for heat_map in pred_lg_heat[j]:
                            # heat_map = nnf.interpolate(heat_map.unsqueeze(0), size=map_size, mode='nearest')
                            heat_map = nnf.interpolate(heat_map.unsqueeze(0).unsqueeze(0),
                                                       size=map_size, mode='bicubic',
                                                       align_corners=False).squeeze(0).squeeze(0)
                            argmax_idx = heat_map.argmax()
                            argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                            pred_lg_ic.append(argmax_idx)

                        pred_lg_ic = torch.tensor(pred_lg_ic).float().to(self.device)
                        pred_lg_ics.append(pred_lg_ic)

                    pred_lg_heat_from_ic = []
                    for lg_idx in range(len(pred_lg_ics)):
                        pred_lg_heat_from_ic.append(self.make_one_heatmap(local_map[lg_idx], pred_lg_ics[lg_idx].detach().cpu().numpy().astype(int)[0]))
                    pred_lg_heat_from_ic = torch.tensor(np.stack(pred_lg_heat_from_ic)).unsqueeze(1).float().to(
                        self.device)
                    pred_sg = F.sigmoid(
                        self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat_from_ic], dim=1)))

                    m = pred_sg.detach().cpu().numpy().copy()[i]
                    m = nnf.interpolate(torch.tensor(m).unsqueeze(0),
                                        size=local_map[i].shape, mode='bicubic',
                                        align_corners=False).squeeze(0).detach().cpu().numpy().copy()



                    for jj in range(3):
                        ax = fig.add_subplot(3, 4, 4*h+2+jj)
                        plt.tight_layout()
                        ax.imshow(env)
                        ax.axis('off')
                        # ax.scatter(local_ic[0, :4, 1], local_ic[0, :4, 0], c='b', s=2, alpha=0.7)
                        # ax.scatter(local_ic[0, 4:, 1], local_ic[0, 4:, 0], c='r', s=2, alpha=0.7)
                        # ax.scatter(local_ic[0, self.sg_idx[jj] + self.obs_len, 1], local_ic[0,  self.sg_idx[jj] + self.obs_len, 0], c='r', s=18, marker='x', alpha=0.7)
                        ax.plot(local_ic[0, :4, 1], local_ic[0, :4, 0],
                                'b--o',
                                c='royalblue',
                                linewidth=1,
                                markersize=1.5,
                                zorder=500,
                                path_effects=[pe.Stroke(linewidth=1, foreground='k'), pe.Normal()], label='GT past')
                        ax.plot(local_ic[0, 4:, 1], local_ic[0, 4:, 0],
                                'r--o',
                                c='darkorange',
                                linewidth=1,
                                markersize=1.5,
                                zorder=500,
                                path_effects=[pe.Stroke(linewidth=1, foreground='k'), pe.Normal()], label='GT future')

                        ax.plot(local_ic[0, self.sg_idx[jj] + self.obs_len, 1], local_ic[0, self.sg_idx[jj] + self.obs_len, 0],
                                'r--x',
                                c='darkorange',
                                linewidth=1,
                                markersize=5,
                                zorder=500,
                                path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()], label='GT past')

                        c = m[jj] / m[jj].max()
                        d = np.stack([1 - c, np.ones_like(c), np.ones_like(c)]).transpose(1, 2, 0)
                        ax.imshow(d, alpha=0.7)



                    ######################
                    # AF
                    import pickle5
                    # with open('C:\dataset\AgentFormer nuscenes k=10\AgentFormer nuscenes k=10/nuscenes_10.pkl', 'rb') as f:
                    with open('D:\crowd/modified_ynet_nu_k102.pkl', 'rb') as f:
                    # with open('C:\dataset/t++\experiments/nuScenes/t_nu10.pkl', 'rb') as f:
                        aa = pickle5.load(f)
                    # gt = aa[1][0, 818]
                    # af_pred = aa[0][:, 818]
                    # our_gt = fut_traj[:, 0, :2]


                    our_gt = fut_traj[:, 0, :2].numpy()
                    idx = np.where(((aa[1][0, :, 0] - our_gt[0]) ** 2).sum(1) < 1)[0][0]
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
                        sns.kdeplot(af_pred_data[:, 0, t, 1], af_pred_data[:, 0, t, 0], bw=None,
                                    ax=ax, shade=True, shade_lowest=False,
                                    color='r', zorder=600, alpha=0.3, legend=True)

                    # ax.scatter(local_ic[0,:,1], local_ic[0,:,0], s=5, c='b')

                    for t in range(10):
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

        total_coll5 = [0] * (lg_num * traj_num)
        total_coll10 = [0] * (lg_num * traj_num)
        total_coll15 = [0] * (lg_num * traj_num)
        total_coll20 = [0] * (lg_num * traj_num)
        total_coll25 = [0] * (lg_num * traj_num)
        total_coll30 = [0] * (lg_num * traj_num)

        sg_total_coll5 = [0] * lg_num
        sg_total_coll10 = [0] * lg_num
        sg_total_coll15 = [0] * lg_num
        sg_total_coll20 = [0] * lg_num
        sg_total_coll25 = [0] * lg_num
        sg_total_coll30 = [0] * lg_num

        n_scene = 0



        all_ade =[]
        all_fde =[]
        sg_ade=[]
        lg_fde=[]
        pred_c = []
        all_pred = []
        all_gt = []
        seq = []

        with torch.no_grad():
            b=0
            while not data_loader.is_epoch_end():
                data = data_loader.next_sample()
                if data is None:
                    continue
                b+=1
                (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
                 maps, local_map, local_ic, local_homo) = data
                batch_size = obs_traj.size(1)
                total_traj += fut_traj.size(1)

                obs_heat_map, sg_heat_map, lg_heat_map = self.make_heatmap(local_ic, local_map)

                self.lg_cvae.forward(obs_heat_map, None, training=False)
                predictions = []
                pred_lg_wcs = []
                pred_sg_wcs = []

                ####### long term goals and the corresponding (deterministic) short term goals ########
                w_priors = []
                for _ in range(lg_num):
                    w_priors.append(self.lg_cvae.prior_latent_space.sample())


                sg_multi_coll5 = []
                sg_multi_coll10 = []
                sg_multi_coll15 = []
                sg_multi_coll20 = []
                sg_multi_coll25 = []
                sg_multi_coll30 = []

                for w_prior in w_priors:
                    # -------- long term goal --------
                    pred_lg_heat = F.sigmoid(self.lg_cvae.sample(self.lg_cvae.unet_enc_feat, w_prior))
                    pred_lg_ics = []
                    pred_lg_wc = []
                    for i in range(batch_size):
                        map_size = local_map[i].shape
                        pred_lg_ic = []
                        for heat_map in pred_lg_heat[i]:
                            # heat_map = nnf.interpolate(heat_map.unsqueeze(0), size=map_size, mode='nearest')
                            heat_map = nnf.interpolate(heat_map.unsqueeze(0).unsqueeze(0),
                                                       size=map_size, mode='bicubic',
                                                       align_corners=False).squeeze(0).squeeze(0)
                            argmax_idx = heat_map.argmax()
                            argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                            pred_lg_ic.append(argmax_idx)

                        pred_lg_ic = torch.tensor(pred_lg_ic).float().to(self.device)

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

                    if generate_heat:
                        pred_lg_heat_from_ic = []
                        for i in range(len(pred_lg_ics)):
                            pred_lg_heat_from_ic.append(self.make_one_heatmap(local_map[i], pred_lg_ics[i][
                                0].detach().cpu().numpy().astype(int)))
                        pred_lg_heat_from_ic = torch.tensor(np.stack(pred_lg_heat_from_ic)).unsqueeze(1).float().to(
                            self.device)
                        pred_sg_heat = F.sigmoid(self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat_from_ic], dim=1)))
                    else:
                        pred_sg_heat = F.sigmoid(self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat], dim=1)))

                    # pred_sg_wc = []
                    # for i in range(batch_size):
                    #     map_size = local_map[i].shape
                    #     pred_sg_ic = []
                    #     for heat_map in pred_sg_heat[i]:
                    #         heat_map = nnf.interpolate(heat_map.unsqueeze(0).unsqueeze(0),
                    #                                    size=map_size, mode='bicubic',
                    #                                    align_corners=False).squeeze(0).squeeze(0)
                    #         argmax_idx = heat_map.flatten().sort().indices[-50:]
                    #
                    #         gara = np.zeros_like(heat_map)
                    #         for r in argmax_idx:
                    #             gara[r // map_size[0], r % map_size[0]] = 1
                    #         plt.imshow(gara)
                    #
                    #         argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                    #         pred_sg_ic.append(argmax_idx)
                    #     pred_sg_ic = torch.tensor(pred_sg_ic).float().to(self.device)
                    #     # ((local_ic[0,[11,15,19]] - pred_sg_ic) ** 2).sum(1).mean()
                    #     back_wc = torch.matmul(
                    #         torch.cat([pred_sg_ic, torch.ones((len(pred_sg_ic), 1)).to(self.device)], dim=1),
                    #         torch.transpose(local_homo[i], 1, 0))
                    #     back_wc /= back_wc[:, 2].unsqueeze(1)
                    #     pred_sg_wc.append(back_wc[:, :2])

                    pred_sg_wc = []
                    for t in range(len(self.sg_idx)):
                        sg_at_this_step = []
                        for s, e in seq_start_end:
                            num_ped = e-s
                            seq_pred_sg_wcs = []
                            for i in range(s,e):
                                map_size = local_map[i].shape
                                pred_sg_ic = []
                                heat_map = nnf.interpolate(pred_sg_heat[i,t].unsqueeze(0).unsqueeze(0),
                                                           size=map_size, mode='bicubic',
                                                           align_corners=False).squeeze(0).squeeze(0)
                                for argmax_idx in heat_map.flatten().sort().indices[-50:]:
                                    argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                                    pred_sg_ic.append(argmax_idx)
                                pred_sg_ic = torch.tensor(pred_sg_ic).float().to(self.device)
                                back_wc = torch.matmul(
                                    torch.cat([pred_sg_ic, torch.ones((len(pred_sg_ic), 1)).to(self.device)], dim=1),
                                    torch.transpose(local_homo[i], 1, 0))
                                back_wc /= back_wc[:, 2].unsqueeze(1)
                                seq_pred_sg_wcs.append(back_wc[:, :2])
                            # check distance btw neighbors within seq_s_e
                            seq_pred_sg_wcs = torch.stack(seq_pred_sg_wcs)
                            final_seq_pred_sg = seq_pred_sg_wcs[:,-1]

                            coll_th = 2.8
                            curr1 = final_seq_pred_sg.repeat(num_ped, 1)
                            curr2 = self.repeat(final_seq_pred_sg, num_ped)
                            dist = torch.sqrt(torch.pow(curr1 - curr2, 2).sum(1)).cpu().numpy()
                            dist = dist.reshape(num_ped, num_ped) + np.eye(num_ped)*100
                            dist[np.triu_indices(num_ped, k=1)] += 100
                            coll_agents = np.where(dist < coll_th)
                            if len(coll_agents[0]) > 0:
                                # print('--------------------------------')
                                # print('before correction: ', len(coll_agents[0]))
                                for c in range(len(coll_agents[0])):
                                    a1_center = final_seq_pred_sg[coll_agents[0][c]]
                                    a2_positions = seq_pred_sg_wcs[coll_agents[1][c]]
                                    dist = torch.sqrt(torch.pow(a2_positions - a1_center, 2).sum(1)).cpu().numpy()
                                    if len(np.where(dist>=coll_th)[0]) > 0:
                                        dist[np.where(dist<coll_th)] +=100
                                        final_seq_pred_sg[coll_agents[1][c]] = a2_positions[dist.argmin()]

                                        curr1 = final_seq_pred_sg.repeat(num_ped, 1)
                                        curr2 = self.repeat(final_seq_pred_sg, num_ped)
                                        dist = torch.sqrt(torch.pow(curr1 - curr2, 2).sum(1)).cpu().numpy()
                                        dist = dist.reshape(num_ped, num_ped) + np.eye(num_ped) * 100
                                        dist[np.triu_indices(num_ped, k=1)] += 100
                                        print('after correction: ', len(np.where(dist < coll_th)[0]))
                                        if len(np.where(dist < coll_th)[0]) == 0 :
                                            break
                                    else:
                                        print('no coll free candidate positions: ', dist.max())
                                        final_seq_pred_sg[coll_agents[1][c]] = a2_positions[dist.argmax()]
                                        # curr1 = final_seq_pred_sg.repeat(num_ped, 1)
                                        # curr2 = self.repeat(final_seq_pred_sg, num_ped)
                                        # dist = torch.sqrt(torch.pow(curr1 - curr2, 2).sum(1)).cpu().numpy()
                                        # dist = dist.reshape(num_ped, num_ped) + np.eye(num_ped) * 100
                                        # dist[np.triu_indices(num_ped, k=1)] += 100
                                        # print('after correction: ', len(np.where(dist < coll_th)[0]))

                            sg_at_this_step.append(final_seq_pred_sg)
                        pred_sg_wc.append(torch.cat(sg_at_this_step)) # bs, 2
                    pred_sg_wc = torch.stack(pred_sg_wc).transpose(1,0) # bs, #sg, 2
                    pred_sg_wcs.append(pred_sg_wc) # for differe w_prior

                    sg_coll5 = 0
                    sg_coll10 = 0
                    sg_coll15 = 0
                    sg_coll20 = 0
                    sg_coll25 = 0
                    sg_coll30 = 0
                    for s, e in seq_start_end:
                        num_ped = e - s
                        if num_ped == 1:
                            continue
                        seq_traj = pred_sg_wc[s:e].transpose(1,0)
                        for i in range(len(seq_traj)):
                            curr1 = seq_traj[i].repeat(num_ped, 1)
                            curr2 = self.repeat(seq_traj[i], num_ped)
                            dist = torch.sqrt(torch.pow(curr1 - curr2, 2).sum(1)).cpu().numpy()
                            dist = dist.reshape(num_ped, num_ped)
                            diff_agent_idx = np.triu_indices(num_ped, k=1)
                            diff_agent_dist = dist[diff_agent_idx]
                            sg_coll5 += (diff_agent_dist < 0.5).sum()
                            sg_coll10 += (diff_agent_dist < 1.0).sum()
                            sg_coll15 += (diff_agent_dist < 1.5).sum()
                            sg_coll20 += (diff_agent_dist < 2.0).sum()
                            sg_coll25 += (diff_agent_dist < 2.5).sum()
                            sg_coll30 += (diff_agent_dist < 2.8).sum()
                    sg_multi_coll5.append(sg_coll5)
                    sg_multi_coll10.append(sg_coll10)
                    sg_multi_coll15.append(sg_coll15)
                    sg_multi_coll20.append(sg_coll20)
                    sg_multi_coll25.append(sg_coll25)
                    sg_multi_coll30.append(sg_coll30)

                    ################


                # a2a collision
                for i in range(lg_num):
                    sg_total_coll5[i] += sg_multi_coll5[i]
                    sg_total_coll10[i] += sg_multi_coll10[i]
                    sg_total_coll15[i] += sg_multi_coll15[i]
                    sg_total_coll20[i] += sg_multi_coll20[i]
                    sg_total_coll25[i] += sg_multi_coll25[i]
                    sg_total_coll30[i] += sg_multi_coll30[i]

                ##### trajectories per long&short goal ####

                # -------- trajectories --------
                (hx, mux, log_varx) \
                    = self.encoderMx(obs_traj_st, seq_start_end)

                p_dist = Normal(mux, torch.sqrt(torch.exp(log_varx)))
                z_priors = []
                for _ in range(traj_num):
                    z_priors.append(p_dist.sample())

                for pred_sg_wc in pred_sg_wcs:
                    for z_prior in z_priors:
                        # -------- trajectories --------
                        # NO TF, pred_goals, z~prior
                        micro_pred = self.decoderMy.make_prediction(
                            seq_start_end,
                            obs_traj_st[-1],
                            obs_traj[-1, :, :2],
                            hx,
                            z_prior,
                            pred_sg_wc,  # goal: (bs, # sg , 2)
                            self.sg_idx
                        )
                        predictions.append(micro_pred)

                multi_coll5 = []
                multi_coll10 = []
                multi_coll15 = []
                multi_coll20 = []
                multi_coll25 = []
                multi_coll30 = []

                ade, fde = [], []
                pred = []
                pix_pred = []
                for vel_pred in predictions:
                    pred_fut_traj=integrate_samples(vel_pred * self.scale, obs_traj[-1, :, :2], dt=self.dt)
                    pred.append(pred_fut_traj)
                    ade.append(displacement_error(
                        pred_fut_traj, fut_traj[:,:,:2], mode='raw'
                    ))
                    fde.append(final_displacement_error(
                        pred_fut_traj[-1], fut_traj[-1,:,:2], mode='raw'
                    ))

                    coll5 = 0
                    coll10 = 0
                    coll15 = 0
                    coll20 = 0
                    coll25 = 0
                    coll30 = 0
                    for s, e in seq_start_end:
                        num_ped = e - s
                        if num_ped == 1:
                            continue
                        seq_traj = pred_fut_traj[:, s:e]
                        for i in range(len(seq_traj)):
                            curr1 = seq_traj[i].repeat(num_ped, 1)
                            curr2 = self.repeat(seq_traj[i], num_ped)
                            dist = torch.sqrt(torch.pow(curr1 - curr2, 2).sum(1)).cpu().numpy()
                            dist = dist.reshape(num_ped, num_ped)
                            diff_agent_idx = np.triu_indices(num_ped, k=1)
                            diff_agent_dist = dist[diff_agent_idx]
                            coll5 += (diff_agent_dist < 0.5).sum()
                            coll10 += (diff_agent_dist < 1.0).sum()
                            coll15 += (diff_agent_dist < 1.5).sum()
                            coll20 += (diff_agent_dist < 2.0).sum()
                            coll25 += (diff_agent_dist < 2.5).sum()
                            coll30 += (diff_agent_dist < 2.8).sum()
                    multi_coll5.append(coll5)
                    multi_coll10.append(coll10)
                    multi_coll15.append(coll15)
                    multi_coll20.append(coll20)
                    multi_coll25.append(coll25)
                    multi_coll30.append(coll30)


                    batch_seq_pix = []
                    for o, (s, e) in enumerate(seq_start_end):
                        for idx in range(s, e):
                            wc = pred_fut_traj.transpose(1, 0)[idx].detach().cpu().numpy()
                            batch_seq_pix.append(maps[o].to_map_points(wc).astype(int))
                    pix_pred.append(np.expand_dims(np.stack(batch_seq_pix), 1))



                all_pred.append(torch.stack(pred).detach().cpu().numpy())
                all_gt.append(fut_traj[:,:,:2].unsqueeze(0).detach().cpu().numpy())
                seq.append(seq_start_end)

                # a2a collision
                for i in range(lg_num * traj_num):
                    total_coll5[i] += multi_coll5[i]
                    total_coll10[i] += multi_coll10[i]
                    total_coll15[i] += multi_coll15[i]
                    total_coll20[i] += multi_coll20[i]
                    total_coll25[i] += multi_coll25[i]
                    total_coll30[i] += multi_coll30[i]

                # a2e collision
                pix_pred = np.concatenate(pix_pred, 1)
                all_maps = []
                for o, (s, e) in enumerate(seq_start_end):
                    m = 1 - maps[o].data / 255
                    m = 1 - m[0] * m[1] * m[2]
                    for _ in range(s, e):
                        all_maps.append(m)
                pred_c.append(compute_ECFL(pix_pred, all_maps))


                # ade / fde
                all_ade.append(torch.stack(ade)) # # sampling, batch size
                all_fde.append(torch.stack(fde))
                sg_ade.append(torch.sqrt(((torch.stack(pred_sg_wcs).permute(0, 2, 1, 3)
                                           - fut_traj[list(self.sg_idx),:,:2].unsqueeze(0).repeat((lg_num,1,1,1)))**2).sum(-1)).sum(1)) # 20, 3, 4, 2
                lg_fde.append(torch.sqrt(((torch.stack(pred_lg_wcs)
                                           - fut_traj[-1,:,:2].unsqueeze(0).repeat((lg_num,1,1)))**2).sum(-1))) # 20, 3, 4, 2


            print("PRED ECFLS: ", np.array(pred_c).mean())

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

            total_coll5=np.array(total_coll5)
            total_coll10=np.array(total_coll10)
            total_coll15=np.array(total_coll15)
            total_coll20=np.array(total_coll20)
            total_coll25=np.array(total_coll25)
            total_coll30=np.array(total_coll30)

            print('total 5: ', np.min(total_coll5, axis=0).mean(), np.mean(total_coll5, axis=0).mean(), np.std(total_coll5, axis=0).mean())
            print('total 10: ', np.min(total_coll10, axis=0).mean(), np.mean(total_coll10, axis=0).mean(), np.std(total_coll10, axis=0).mean())
            print('total 15: ', np.min(total_coll15, axis=0).mean(), np.mean(total_coll15, axis=0).mean(), np.std(total_coll15, axis=0).mean())
            print('total 20: ', np.min(total_coll20, axis=0).mean(), np.mean(total_coll20, axis=0).mean(), np.std(total_coll20, axis=0).mean())
            print('total 25: ', np.min(total_coll25, axis=0).mean(), np.mean(total_coll25, axis=0).mean(), np.std(total_coll25, axis=0).mean())
            print('total 30: ', np.min(total_coll30, axis=0).mean(), np.mean(total_coll30, axis=0).mean(), np.std(total_coll30, axis=0).mean())
            print('=========================================== sg collision')
            print('total 5: ', np.min(sg_total_coll5, axis=0).mean(), np.mean(sg_total_coll5, axis=0).mean(),
                  np.std(sg_total_coll5, axis=0).mean())
            print('total 10: ', np.min(sg_total_coll10, axis=0).mean(), np.mean(sg_total_coll10, axis=0).mean(),
                  np.std(sg_total_coll10, axis=0).mean())
            print('total 15: ', np.min(sg_total_coll15, axis=0).mean(), np.mean(sg_total_coll15, axis=0).mean(),
                  np.std(sg_total_coll15, axis=0).mean())
            print('total 20: ', np.min(sg_total_coll20, axis=0).mean(), np.mean(sg_total_coll20, axis=0).mean(),
                  np.std(sg_total_coll20, axis=0).mean())
            print('total 25: ', np.min(sg_total_coll25, axis=0).mean(), np.mean(sg_total_coll25, axis=0).mean(),
                  np.std(sg_total_coll25, axis=0).mean())
            print('total 30: ', np.min(sg_total_coll30, axis=0).mean(), np.mean(sg_total_coll30, axis=0).mean(),
                  np.std(sg_total_coll30, axis=0).mean())

            print(n_scene)

        import pickle
        # # sampling, # data, pred_len, 2
        data = [np.concatenate(all_pred, -2).transpose(0,2,1,3), np.concatenate(all_gt, -2).transpose(0,2,1,3), np.concatenate(seq)]
        with open('./nu_gen/' + self.save_path.split('nu.traj_')[1] + '_pred' + str(traj_num*lg_num) + '.pkl', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return ade_min, fde_min, \
               ade_avg, fde_avg, \
               ade_std, fde_std, \
               sg_ade_min, sg_ade_avg, sg_ade_std, \
               lg_fde_min, lg_fde_avg, lg_fde_std




    def make_ecfl(self, data_loader, lg_num=5, traj_num=4, generate_heat=True):
        self.set_mode(train=False)

        pred_c = []
        gt_c = []
        n_sample = 0
        with torch.no_grad():
            b=0
            while not data_loader.is_epoch_end():
                data = data_loader.next_sample()
                if data is None:
                    continue
                b+=1
                (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
                 maps, local_map, local_ic, local_homo) = data
                batch_size = obs_traj.size(1)
                obs_heat_map, sg_heat_map, lg_heat_map = self.make_heatmap(local_ic, local_map)


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
                    pred_lg_ics = []
                    pred_lg_wc = []
                    for i in range(batch_size):
                        map_size = local_map[i].shape
                        pred_lg_ic = []
                        for heat_map in pred_lg_heat[i]:
                            # heat_map = nnf.interpolate(heat_map.unsqueeze(0), size=map_size, mode='nearest')
                            heat_map = nnf.interpolate(heat_map.unsqueeze(0).unsqueeze(0),
                                                       size=map_size, mode='bicubic',
                                                       align_corners=False).squeeze(0).squeeze(0)
                            argmax_idx = heat_map.argmax()
                            argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                            pred_lg_ic.append(argmax_idx)

                        pred_lg_ic = torch.tensor(pred_lg_ic).float().to(self.device)

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

                    if generate_heat:
                        pred_lg_heat_from_ic = []
                        for i in range(len(pred_lg_ics)):
                            pred_lg_heat_from_ic.append(self.make_one_heatmap(local_map[i], pred_lg_ics[i][
                                0].detach().cpu().numpy().astype(int)))
                        pred_lg_heat_from_ic = torch.tensor(np.stack(pred_lg_heat_from_ic)).unsqueeze(1).float().to(
                            self.device)
                        pred_sg_heat = F.sigmoid(self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat_from_ic], dim=1)))
                    else:
                        pred_sg_heat = F.sigmoid(self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat], dim=1)))

                    pred_sg_wc = []
                    for i in range(batch_size):
                        map_size = local_map[i].shape
                        pred_sg_ic = []
                        for heat_map in pred_sg_heat[i]:
                            heat_map = nnf.interpolate(heat_map.unsqueeze(0).unsqueeze(0),
                                                       size=map_size, mode='bicubic',
                                                       align_corners=False).squeeze(0).squeeze(0)
                            argmax_idx = heat_map.argmax()
                            argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                            pred_sg_ic.append(argmax_idx)
                        pred_sg_ic = torch.tensor(pred_sg_ic).float().to(self.device)
                        # ((local_ic[0,[11,15,19]] - pred_sg_ic) ** 2).sum(1).mean()
                        back_wc = torch.matmul(
                            torch.cat([pred_sg_ic, torch.ones((len(pred_sg_ic), 1)).to(self.device)], dim=1),
                            torch.transpose(local_homo[i], 1, 0))
                        back_wc /= back_wc[:, 2].unsqueeze(1)
                        pred_sg_wc.append(back_wc[:, :2])
                        # ((back_wc - fut_traj[[3, 7, 11], 0, :2]) ** 2).sum(1).mean()
                    pred_sg_wc = torch.stack(pred_sg_wc)
                    pred_sg_wcs.append(pred_sg_wc)

                    ################


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
                    pred_fut_traj=integrate_samples(dist.rsample() * self.scale, obs_traj[-1, :, :2], dt=self.dt)

                    # idx = 1
                    # pixel = maps[0].to_map_points(pred_fut_traj[:,idx])
                    # plt.imshow(maps[0].data.transpose(1, 2, 0))
                    # map_traj = maps[0].to_map_points(fut_traj[:, idx, :2])
                    # plt.scatter(map_traj[:, 1], map_traj[:, 0], s=1, c='w')
                    # plt.scatter(pixel[:, 1], pixel[:, 0], s=1, c='b')

                    batch_seq_pix = []
                    for o, (s,e) in enumerate(seq_start_end):
                        for idx in range(s,e):
                            wc = pred_fut_traj.transpose(1, 0)[idx].detach().cpu().numpy()
                            batch_seq_pix.append(maps[o].to_map_points(wc).astype(int))
                    pred.append(np.expand_dims(np.stack(batch_seq_pix), 1))
                pred = np.concatenate(pred, 1)

                all_maps=[]
                for o, (s,e) in enumerate(seq_start_end):
                    m = 1 - maps[o].data / 255
                    m = 1 - m[0] * m[1] * m[2]
                    for _ in range(s,e):
                        all_maps.append(m)

                pred_c.append(compute_ECFL(pred,all_maps))

                gt_map_pt = []
                gt = fut_traj[:,:,:2].transpose(1,0).unsqueeze(1).detach().cpu().numpy()
                for o, (s, e) in enumerate(seq_start_end):
                    for idx in range(s, e):
                        gt_map_pt.append(maps[o].to_map_points(gt[idx]).astype(int))

                gt_c.append(compute_ECFL(np.stack(gt_map_pt).astype(int), all_maps))
                n_sample += gt.shape[0]

        print('N SAMPLE: ', n_sample)
        print("GT ECFLS: ", np.array(gt_c).mean())
        print("PRED ECFLS: ", np.array(pred_c).mean())



    def check_coll(self, data_loader):
        self.set_mode(train=False)

        lg_num = 20
        traj_num=1
        generate_heat = True
        root = 'C:/dataset/AgentFormer/data/datasets/nuscenes_pred/datasets/nuscenes_pred'
        with torch.no_grad():
            # seq_index, frame = data_loader.get_seq_and_frame(idx)


            idx=133
            idx=595
            idx=298

            idx=413
            idx=534
            idx=1572
            idx=1871
            idx=1999
            idx=2896
            idx=2886
            idx=2989
            idx=2775
            data_loader.index = idx
            data = data_loader.next_sample()
            (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
             maps, local_map, local_ic, local_homo) = data  #  obs_traj= (past step 4, batch size, 6 state)

            s, e = seq_start_end[0]
            batch_size = e-s
            num_ped = batch_size
            print(num_ped)

            seq_traj = fut_traj[:, :, :2]
            for i in range(len(seq_traj)):
                curr1 = seq_traj[i].repeat(num_ped, 1)
                curr2 = self.repeat(seq_traj[i], num_ped)
                dist = torch.sqrt(torch.pow(curr1 - curr2, 2).sum(1))
                dist = dist.reshape(num_ped, num_ped).cpu().numpy()
                dist[np.diag_indices(num_ped)] +=100
                # diff_agent_idx = np.triu_indices(num_ped, k=1)
                # diff_agent_dist = dist[diff_agent_idx]
                # if dist.min() < 0.55:
                    # print(idx)
                print(i)
                print(dist.min())
                print(dist.argmin()//num_ped, dist.argmin()%num_ped)
                print('------------------')

            map_traj = []
            for i in range(s,e):
                wc = fut_traj[:,:,:2].transpose(1, 0)[i].detach().cpu().numpy()
                map_traj.append(maps[0].to_map_points(wc).astype(int))
            map_traj = np.array(map_traj)
            # m = maps[0].data.transpose(1,2,0)
            m = 1 - maps[0].data / 255
            m[0][np.where(m[1] == 0)] = 0.3
            m[0][np.where(m[2] == 0)] = 0.6
            m = m[0]

            env = 1-np.stack([m, m, m], axis=2)
            plt.imshow(env)

            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            for i in range(s,e):
                plt.scatter(map_traj[i,:,1], map_traj[i,:,0], c=colors[i], s=1)
                plt.scatter(map_traj[i,0,1], map_traj[i,0,0], c=colors[i], s=15, marker='x')
            print(num_ped)


            neighbor_idx = (0,1)
            neighbor_idx = (1,2)
            for i in range(len(seq_traj)):
                curr1 = seq_traj[i].repeat(num_ped, 1)
                curr2 = self.repeat(seq_traj[i], num_ped)
                dist = torch.sqrt(torch.pow(curr1 - curr2, 2).sum(1))
                dist = dist.reshape(num_ped, num_ped).cpu().numpy()
                print(dist[neighbor_idx])



            obs_heat_map, sg_heat_map, lg_heat_map = self.make_heatmap(local_ic, local_map)

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
                pred_lg_ics = []
                pred_lg_wc = []
                for i in range(batch_size):
                    map_size = local_map[i].shape
                    pred_lg_ic = []
                    for heat_map in pred_lg_heat[i]:
                        # heat_map = nnf.interpolate(heat_map.unsqueeze(0), size=map_size, mode='nearest')
                        heat_map = nnf.interpolate(heat_map.unsqueeze(0).unsqueeze(0),
                                                   size=map_size, mode='bicubic',
                                                   align_corners=False).squeeze(0).squeeze(0)
                        argmax_idx = heat_map.argmax()
                        argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                        pred_lg_ic.append(argmax_idx)

                    pred_lg_ic = torch.tensor(pred_lg_ic).float().to(self.device)

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

                if generate_heat:
                    pred_lg_heat_from_ic = []
                    for i in range(len(pred_lg_ics)):
                        pred_lg_heat_from_ic.append(self.make_one_heatmap(local_map[i], pred_lg_ics[i][
                            0].detach().cpu().numpy().astype(int)))
                    pred_lg_heat_from_ic = torch.tensor(np.stack(pred_lg_heat_from_ic)).unsqueeze(1).float().to(
                        self.device)
                    pred_sg_heat = F.sigmoid(
                        self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat_from_ic], dim=1)))
                else:
                    pred_sg_heat = F.sigmoid(
                        self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat], dim=1)))

                pred_sg_wc = []
                for i in range(batch_size):
                    map_size = local_map[i].shape
                    pred_sg_ic = []
                    for heat_map in pred_sg_heat[i]:
                        heat_map = nnf.interpolate(heat_map.unsqueeze(0).unsqueeze(0),
                                                   size=map_size, mode='bicubic',
                                                   align_corners=False).squeeze(0).squeeze(0)
                        argmax_idx = heat_map.argmax()
                        argmax_idx = [argmax_idx // map_size[0], argmax_idx % map_size[0]]
                        pred_sg_ic.append(argmax_idx)
                    pred_sg_ic = torch.tensor(pred_sg_ic).float().to(self.device)
                    # ((local_ic[0,[11,15,19]] - pred_sg_ic) ** 2).sum(1).mean()
                    back_wc = torch.matmul(
                        torch.cat([pred_sg_ic, torch.ones((len(pred_sg_ic), 1)).to(self.device)], dim=1),
                        torch.transpose(local_homo[i], 1, 0))
                    back_wc /= back_wc[:, 2].unsqueeze(1)
                    pred_sg_wc.append(back_wc[:, :2])
                    # ((back_wc - fut_traj[[3, 7, 11], 0, :2]) ** 2).sum(1).mean()
                pred_sg_wc = torch.stack(pred_sg_wc)
                pred_sg_wcs.append(pred_sg_wc)

                ################

            ##### trajectories per long&short goal ####

            # -------- trajectories --------
            (hx, mux, log_varx) \
                = self.encoderMx(obs_traj_st, seq_start_end)
                # = self.encoderMx(obs_traj_st, seq_start_end, self.lg_cvae.unet_enc_feat, local_homo)

            p_dist = Normal(mux, torch.sqrt(torch.exp(log_varx)))
            z_priors = []
            for _ in range(traj_num):
                z_priors.append(p_dist.sample())

            for pred_sg_wc in pred_sg_wcs:
                for z_prior in z_priors:
                    # -------- trajectories --------
                    # NO TF, pred_goals, z~prior
                    fut_rel_pos_dist_prior = self.decoderMy(
                        seq_start_end,
                        obs_traj_st[-1],
                        obs_traj[-1, :, :2],
                        hx,
                        z_prior,
                        pred_sg_wc,  # goal
                        self.sg_idx
                    )
                    fut_rel_pos_dists.append(fut_rel_pos_dist_prior)


            pred_data = []
            for dist in fut_rel_pos_dists:
                pred_fut_traj = integrate_samples(dist.rsample() * self.scale, obs_traj[-1, :, :2], dt=self.dt)
                pred_data.append(pred_fut_traj)

                # for s, e in seq_start_end:
                #     num_ped = e - s
                #     if num_ped == 1:
                #         continue
                #     seq_traj = pred_fut_traj[:, s:e]
                #     for i in range(len(seq_traj)):
                #         curr1 = seq_traj[i].repeat(num_ped, 1)
                #         curr2 = self.repeat(seq_traj[i], num_ped)
                #         dist = torch.sqrt(torch.pow(curr1 - curr2, 2).sum(1)).cpu().numpy()
                #         dist = dist.reshape(num_ped, num_ped)
                #         print(dist[neighbor_idx])


            pred_data = torch.stack(pred_data).numpy().transpose(0,2,1,3)

            # ---------- plot gif
            wc_traj = torch.cat([obs_traj[:,:,:2], fut_traj[:,:,:2]], 0).numpy().transpose(1,0,2)
            n_agent = wc_traj.shape[0]
            ic_traj=[]
            for wc in wc_traj:
                ic_traj.append(maps[0].to_map_points(wc).astype(int))
            ic_traj = np.array(ic_traj)
            ic_pred = []
            for i in range(traj_num * lg_num):
                temp = []
                for wc in pred_data[i]:
                    temp.append(maps[0].to_map_points(wc).astype(int))
                # temp = np.array(temp)
                ic_pred.append(temp)
            ic_pred = np.array(ic_pred)
            ic_traj = ic_traj[:,:,[1,0]]
            ic_pred = ic_pred[:,:,:,[1,0]]
            env = 1-np.stack([m, m, m], axis=2)

            '''
            plt.imshow(m)
            plt.scatter(ic_pred[0,a,:, 0], ic_pred[0, a, :, 1], s=1, c='g')           
            
            plt.scatter(ic_traj[a,:4, 0], ic_traj[a, :4, 1], s=1, c='b')
            plt.scatter(ic_traj[a, 4:, 0], ic_traj[a, 4:, 1], s=1, c='r')
            '''


            # ========================================================


            num_ped = n_agent
            for pred_idx in range(20):
                seq_traj = torch.tensor(pred_data[pred_idx].transpose(1, 0, 2))
                for i in range(len(seq_traj)):
                    curr1 = seq_traj[i].repeat(num_ped, 1)
                    curr2 = self.repeat(seq_traj[i], num_ped)
                    dist = torch.sqrt(torch.pow(curr1 - curr2, 2).sum(1)).cpu().numpy()
                    dist = dist.reshape(num_ped, num_ped)
                    dist[np.diag_indices(num_ped)] += 100
                    if (dist < 1.5).sum()>0:
                        print('=======')
                        print(pred_idx)
                        print('t: ', i)
                        print(np.where(dist<1.5))
                        print(dist[np.where(dist<1.5)])



            def pre_gif(pred_idx):
                def init():
                    ax.imshow(env)
                    ax.axis('off')

                def update_dot(num_t):
                    print(num_t)
                    ax.imshow(env)
                    for agent_idx in range(n_agent):
                        if num_t >= 4:
                            ln_pred[agent_idx].set_data(ic_pred[pred_idx, agent_idx, :num_t-3, 0],
                                                        ic_pred[pred_idx, agent_idx, :num_t-3, 1])
                        ln_gt[agent_idx].set_data(ic_traj[agent_idx, :num_t+1, 0],
                                                  ic_traj[agent_idx, :num_t+1, 1])

                fig, ax = plt.subplots(figsize=(7, 7))
                ax.axis('off')
                # ax.set_title('sampling number', str(pred_idx), fontsize=9)
                fig.tight_layout()
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

                ln_pred = []
                ln_gt = []
                for s in range(n_agent):
                    c = str(colors[s%len(colors)])
                    ln_pred.append(
                        ax.plot([], [], c + '.-', alpha=0.5, linewidth=1, markersize=2)[0])
                    ln_gt.append(
                        ax.plot([], [], c + '--', alpha=0.5, linewidth=1)[0])

                return fig, update_dot


            def allpre_gif(agent_idx):
                def init():
                    ax.imshow(env)
                    ax.axis('off')

                def update_dot(num_t):
                    print(num_t)
                    ax.imshow(env)
                    if num_t >= 4:
                        for pred_idx in range(0,10):
                                ln_pred[pred_idx].set_data(ic_pred[pred_idx, agent_idx, :num_t-3, 0],
                                                            ic_pred[pred_idx, agent_idx, :num_t-3, 1])
                    if num_t <= 4:
                        ln_gt[0].set_data(ic_traj[agent_idx, :num_t+1, 0],
                                                  ic_traj[agent_idx, :num_t+1, 1])

                fig, ax = plt.subplots(figsize=(14, 14))
                ax.axis('off')
                # ax.set_title('sampling number', str(pred_idx), fontsize=9)
                fig.tight_layout()
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

                ln_pred = []
                ln_gt = [ax.plot([], [], 'b--', alpha=0.7, linewidth=2)[0]]
                for s in range(10):
                    c = str(colors[s%len(colors)])
                    ln_pred.append(
                        ax.plot([], [],  'r--', alpha=0.7, linewidth=2, markersize=2)[0])
                return fig, update_dot

            for agent_idx in range(n_agent):
                fig, update_dot = pre_gif(pred_idx)
                # fig, update_dot = allpre_gif(2)
                ani = FuncAnimation(fig, update_dot, frames=16, interval=1, init_func=init())
                plt.close(fig)
                # ani_path = os.path.join(root, 'gif', 'nopool')
                ani_path = os.path.join(root, 'gif', 'pool')
                ani.save(os.path.join(ani_path, str(idx)+ '_runid83_sampling' + str(pred_idx)+".gif"), fps=4)
                # ani.save(os.path.join(ani_path, str(idx)+ '_agent2.gif'), fps=4)



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

        map_ae_path = 'mapae.nu_lr_0.001_a_0.25_r_2.0_run_3'
        map_ae_path = os.path.join('ckpts', map_ae_path, 'iter_11984_sg_unet.pt')
        self.save_path = traj['ckpt_dir']

        if self.device == 'cuda':
            self.map_ae = torch.load(map_ae_path)
        else:
            self.map_ae = torch.load(map_ae_path, map_location='cpu')
        print(">>>>>>>>> map Init: ", map_ae_path)


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
