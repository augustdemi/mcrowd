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
from data.nuscenes.config import Config
from data.nuscenes_dataloader import data_generator
from unet.unet import Unet
import numpy as np
import visdom
import torch.nn.functional as F
import torch.nn.functional as nnf


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



class Solver(object):

    ####
    def __init__(self, args):

        self.args = args

        self.name = '%s_lr_%s_a_%s_r_%s_aug_%s_num_sg_%s' % \
                    (args.dataset_name, args.lr_VAE, args.alpha, args.gamma, args.aug, args.load_e)

        # to be appended by run_id

        # self.use_cuda = args.cuda and torch.cuda.is_available()
        self.fb = args.fb
        self.anneal_epoch = args.anneal_epoch
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.aug = args.aug
        self.device = args.device
        self.temp=1.99
        self.dt=0.1
        self.eps=1e-9
        self.ll_prior_w =args.ll_prior_w
        self.sg_idx = np.array(range(40))
        num_sg = args.load_e
        self.sg_idx = np.flip(39-self.sg_idx[::(40//num_sg)])
        print('>>>>> sg location: ', self.sg_idx + 1)


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

        # visdom setup
        self.viz_on = args.viz_on
        if self.viz_on:
            self.win_id = dict(
                recon='win_recon', test_total_loss='win_test_total_loss',
                lg_fde_min='win_lg_fde_min', lg_fde_avg='win_lg_fde_avg', lg_fde_std='win_lg_fde_std',
                sg_recon='win_sg_recon', test_sg_recon='win_test_sg_recon',
                sg_ade_min='win_sg_ade_min', sg_ade_avg='win_sg_ade_avg', sg_ade_std='win_sg_ade_std',
            )
            self.line_gather = DataGather(
                'iter',
                'test_total_loss',
                'lg_fde_min', 'lg_fde_avg', 'lg_fde_std',
                'sg_recon', 'test_sg_recon',
                'sg_ade_min', 'sg_ade_avg', 'sg_ade_std',
            )


            self.viz_port = args.viz_port  # port number, eg, 8097
            self.viz = visdom.Visdom(port=self.viz_port)
            self.viz_ll_iter = args.viz_ll_iter
            self.viz_la_iter = args.viz_la_iter

            self.viz_init()

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


        #### create a new model or load a previously saved model

        self.ckpt_load_iter = args.ckpt_load_iter

        self.obs_len = 20
        self.pred_len = args.pred_len
        self.num_layers = args.num_layers
        self.decoder_h_dim = args.decoder_h_dim


        # lg_cvae_path = 'ki.lgcvae_enc_block_1_fcomb_block_2_wD_10_lr_0.0001_lg_klw_1.0_a_0.25_r_2.0_fb_2.5_anneal_e_10_aug_1_llprior_1.0_run_0'
        # lg_cvae_path = os.path.join('ckpts', lg_cvae_path, 'iter_20250_lg_cvae.pt')
        #
        # if self.device == 'cuda':
        #     self.lg_cvae = torch.load(lg_cvae_path)
        # else:
        #     self.lg_cvae = torch.load(lg_cvae_path, map_location='cpu')
        # print(">>>>>>>>> Init: ", lg_cvae_path)

        if self.ckpt_load_iter == 0 or args.dataset_name =='all':  # create a new model
            num_filters = [32, 32, 64, 64, 64, 128]
            # input = env + 8 past + lg / output = env + sg(including lg)
            self.sg_unet = Unet(input_channels=3, num_classes=len(self.sg_idx), num_filters=num_filters,
                             apply_last_layer=True, padding=True).to(self.device)


        else:  # load a previously saved model
            print('Loading saved models (iter: %d)...' % self.ckpt_load_iter)
            self.load_checkpoint()

            print('...done')


        # get VAE parameters
        vae_params = \
            list(self.sg_unet.parameters())

        # create optimizers
        self.optim_vae = optim.Adam(
            vae_params,
            lr=self.lr_VAE,
            betas=[self.beta1_VAE, self.beta2_VAE]
        )
        # self.lg_optimizer = torch.optim.Adam(, lr=self., weight_decay=0)

        # prepare dataloader (iterable)
        print('Start loading data...')

        if self.ckpt_load_iter != self.max_iter:
            print("Initializing train dataset")
            _, self.train_loader = data_loader(self.args, args.dataset_dir, 'train', shuffle=True)
            print("Initializing val dataset")
            _, self.val_loader = data_loader(self.args, args.dataset_dir, 'test', shuffle=False)
            # _, self.val_loader = _, self.train_loader


            print(
                'There are {} iterations per epoch'.format(len(self.train_loader.dataset) / args.batch_size)
            )
        print('...done')




    def make_heatmap(self, local_ic, local_map, aug=False):
        heat_maps=[]
        down_size=256
        for i in range(len(local_ic)):
            '''
            plt.imshow(local_map[i])
            plt.scatter(local_ic[i,:4,1], local_ic[i,:4,0], s=1, c='b')
            plt.scatter(local_ic[i,4:,1], local_ic[i,4:,0], s=1, c='g')
            '''
            map_size = local_map[i].shape[0]
            env = cv2.resize(local_map[i], dsize=(down_size, down_size))
            ohm = [env/3]
            heat_map_traj = np.zeros_like(local_map[i])
            heat_map_traj[local_ic[i, :self.obs_len, 0], local_ic[i, :self.obs_len, 1]] = 100

            if map_size > 1000:
                heat_map_traj = cv2.resize(ndimage.filters.gaussian_filter(heat_map_traj, sigma=2),
                                           dsize=((map_size + down_size) // 2, (map_size + down_size) // 2))
                heat_map_traj = heat_map_traj / heat_map_traj.sum()
            heat_map_traj = cv2.resize(ndimage.filters.gaussian_filter(heat_map_traj, sigma=2),
                                       dsize=(down_size, down_size))
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
            for j in (self.sg_idx + self.obs_len):
                heat_map_traj = np.zeros_like(local_map[i])
                heat_map_traj[local_ic[i, j, 0], local_ic[i, j, 1]] = 1000
                if map_size > 1000:
                    heat_map_traj = cv2.resize(ndimage.filters.gaussian_filter(heat_map_traj, sigma=2),
                                               dsize=((map_size + down_size) // 2, (map_size + down_size) // 2))
                heat_map_traj = cv2.resize(ndimage.filters.gaussian_filter(heat_map_traj, sigma=2),
                                           dsize=(down_size, down_size))
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
    def train(self):
        self.set_mode(train=True)
        data_loader = self.train_loader

        self.N = len(data_loader.dataset)
        iterator = iter(data_loader)

        iter_per_epoch = len(iterator)
        start_iter = self.ckpt_load_iter + 1
        epoch = int(start_iter / iter_per_epoch)

        for iteration in range(start_iter, self.max_iter + 1):

            # reset data iterators for each epoch
            if iteration % iter_per_epoch == 0:
                print('==== epoch %d done ====' % epoch)
                epoch +=1
                iterator = iter(data_loader)

            # ============================================
            #          TRAIN THE VAE (ENC & DEC)
            # ============================================

            (obs_traj, fut_traj, obs_traj_st, fut_traj_st, seq_start_end,
             videos, classes, global_map, homo,
             local_map, local_ic, local_homo) = next(iterator)
            batch_size = obs_traj.size(1) #=sum(seq_start_end[:,1] - seq_start_end[:,0])

            '''
            map_idx = np.where([i in range(elt[0], elt[1]) for elt in seq_start_end])[0][0]
            plt.imshow(maps[map_idx].data.transpose(2,1,0))
            all_traj = torch.cat([obs_traj[:,i,:2], fut_traj[:,i,:2]], dim=0)
            all_traj = maps[map_idx].to_map_points(all_traj*cfg['traj_scale'])
            plt.scatter(all_traj[:4,0], all_traj[:4,1], s=1, c='b')
            plt.scatter(all_traj[4:,0], all_traj[4:,1], s=1, c='g')
            '''
            obs_heat_map, sg_heat_map, lg_heat_map =  self.make_heatmap(local_ic, local_map, aug=self.aug)

            # -------- short term goal --------
            # obs_lg_heat = torch.cat([obs_heat_map, lg_heat_map[:,-1].unsqueeze(1)], dim=1)
            recon_sg_heat = self.sg_unet.forward(torch.cat([obs_heat_map, lg_heat_map], dim=1))
            recon_sg_heat = F.sigmoid(recon_sg_heat)
            normalized_recon_sg_heat = []
            for i in range(len(self.sg_idx)):
                sg_map = recon_sg_heat[:,i]
                normalized_recon_sg_heat.append(F.normalize(sg_map.view((sg_map.shape[0], -1)), p=1))
            recon_sg_heat = torch.stack(normalized_recon_sg_heat, dim=1)
            sg_heat_map= sg_heat_map.view(sg_heat_map.shape[0], len(self.sg_idx), -1)


            sg_recon_loss = - (
            self.alpha * sg_heat_map * torch.log(recon_sg_heat + self.eps) * ((1 - recon_sg_heat) ** self.gamma) \
            + (1 - self.alpha) * (1 - sg_heat_map) * torch.log(1 - recon_sg_heat + self.eps) * (
                recon_sg_heat ** self.gamma)).sum().div(batch_size)

            loss = sg_recon_loss

            self.optim_vae.zero_grad()
            loss.backward()
            self.optim_vae.step()


            # save model parameters
            if iteration % self.ckpt_save_iter == 0:
                self.save_checkpoint(iteration)

            # (visdom) insert current line stats
            if self.viz_on and (iteration % self.viz_ll_iter == 0):
                lg_fde_min, lg_fde_avg, lg_fde_std, test_lg_recon, test_lg_kl, \
                        test_sg_recon_loss, sg_ade_min, sg_ade_avg, sg_ade_std = self.evaluate_dist(self.val_loader, loss=True)
                test_total_loss = test_sg_recon_loss
                self.line_gather.insert(iter=iteration,
                                        lg_fde_min=lg_fde_min,
                                        lg_fde_avg=lg_fde_avg,
                                        lg_fde_std=lg_fde_std,
                                        test_total_loss=test_total_loss.item(),
                                        sg_recon=sg_recon_loss.item(),
                                        test_sg_recon=test_sg_recon_loss.item(),
                                        sg_ade_min=sg_ade_min,
                                        sg_ade_avg=sg_ade_avg,
                                        sg_ade_std=sg_ade_std,
                                        )

                prn_str = ('[iter_%d (epoch_%d)] VAE Loss: %.3f '
                          ) % \
                          (iteration, epoch,
                           loss.item(),
                           )

                print(prn_str)


                if self.record_file:
                    record = open(self.record_file, 'a')
                    record.write('%s\n' % (prn_str,))
                    record.close()


            # (visdom) visualize line stats (then flush out)
            if self.viz_on and (iteration % self.viz_la_iter == 0):
                self.visualize_line()
                self.line_gather.flush()



    def evaluate_dist(self, data_loader, loss=False):
        self.set_mode(train=False)
        total_traj = 0

        sg_recon=0
        sg_ade = []
        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_st, fut_traj_st, seq_start_end,
                 videos, classes, global_map, homo,
                 local_map, local_ic, local_homo) = batch
                batch_size = obs_traj.size(1)
                total_traj += fut_traj.size(1)

                obs_heat_map, sg_heat_map, lg_heat_map = self.make_heatmap(local_ic, local_map)

                # self.lg_cvae.forward(obs_heat_map, None, training=False)
                pred_sg_wcs = []
                for _ in range(5):
                    # -------- long term goal --------
                    # pred_lg_heat = F.sigmoid(self.lg_cvae.sample(testing=True))
                    pred_lg_heat = lg_heat_map

                    pred_lg_wc = []
                    pred_lg_ics = []
                    for i in range(batch_size):
                        map_size = local_map[i].shape
                        pred_lg_ic = []
                        for heat_map in pred_lg_heat[i]:
                            # heat_map = nnf.interpolate(heat_map.unsqueeze(0), size=map_size, mode='nearest')
                            heat_map = nnf.interpolate(heat_map.unsqueeze(0).unsqueeze(0),
                                                       size=map_size, mode='bicubic',
                                                       align_corners=False).squeeze(0).squeeze(0)
                            argmax_idx = heat_map.argmax()
                            argmax_idx = [argmax_idx//map_size[0], argmax_idx%map_size[0]]
                            pred_lg_ic.append(argmax_idx)

                        pred_lg_ic = torch.tensor(pred_lg_ic).float().to(self.device)
                        pred_lg_ics.append(pred_lg_ic)

                    # -------- short term goal --------
                    pred_lg_heat_from_ic = []
                    for i in range(len(pred_lg_ics)):
                        pred_lg_heat_from_ic.append(self.make_one_heatmap(local_map[i], pred_lg_ics[i][0].detach().cpu().numpy().astype(int)))
                    pred_lg_heat_from_ic = torch.tensor(np.stack(pred_lg_heat_from_ic)).unsqueeze(1).float().to(self.device)

                    pred_sg_heat = F.sigmoid(self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat_from_ic], dim=1)))

                    pred_sg_wc = []
                    for i in range(batch_size):
                        map_size = local_map[i].shape
                        pred_sg_ic = []
                        for heat_map in pred_sg_heat[i]:
                            heat_map = nnf.interpolate(heat_map.unsqueeze(0).unsqueeze(0),
                                                       size=map_size, mode='bicubic',
                                                       align_corners=False).squeeze(0).squeeze(0)
                            argmax_idx = heat_map.argmax()
                            argmax_idx = [argmax_idx//map_size[0], argmax_idx%map_size[0]]
                            pred_sg_ic.append(argmax_idx)
                        pred_sg_ic = torch.tensor(pred_sg_ic).float().to(self.device)
                        # ((local_ic[0,[11,15,19]] - pred_sg_ic) ** 2).sum(1).mean()
                        back_wc = torch.matmul(
                            torch.cat([pred_sg_ic, torch.ones((len(pred_sg_ic), 1)).to(self.device)], dim=1),
                            torch.transpose(torch.tensor(local_homo[i]).float().to(self.device), 1, 0))
                        back_wc /= back_wc[:, 2].clone().unsqueeze(1)
                        pred_sg_wc.append(back_wc[:, :2])
                        # ((back_wc - fut_traj[[3, 7, 11], 0, :2]) ** 2).sum(1).mean()
                    pred_sg_wc = torch.stack(pred_sg_wc)
                    pred_sg_wcs.append(pred_sg_wc)


                if loss:
                    normalized_recon_sg_heat = []
                    for i in range(len(self.sg_idx)):
                        sg_map = pred_sg_heat[:, i]
                        normalized_recon_sg_heat.append(F.normalize(sg_map.view((sg_map.shape[0], -1)), p=1))
                    pred_sg_heat = torch.stack(normalized_recon_sg_heat, dim=1)
                    sg_heat_map = sg_heat_map.view(sg_heat_map.shape[0], len(self.sg_idx), -1)

                    sg_recon += - (self.alpha * sg_heat_map * torch.log(pred_sg_heat + self.eps) * ((1 - pred_sg_heat) ** self.gamma) \
                                    + (1 - self.alpha) * (1 - sg_heat_map) * torch.log(1 - pred_sg_heat + self.eps) * (
                                       pred_sg_heat ** self.gamma)).sum().div(batch_size)

                sg_ade.append(torch.sqrt(((torch.stack(pred_sg_wcs).permute(0, 2, 1, 3)
                                           - fut_traj[list(self.sg_idx), :, :2].unsqueeze(0).repeat((5, 1, 1, 1))) ** 2).sum(-1)).sum(1))

            sg_ade=torch.cat(sg_ade, dim=1).cpu().numpy()


            sg_ade_min = np.min(sg_ade, axis=0).mean()/len(self.sg_idx)
            sg_ade_avg = np.mean(sg_ade, axis=0).mean()/len(self.sg_idx)
            sg_ade_std = np.std(sg_ade, axis=0).mean()/len(self.sg_idx)

        self.set_mode(train=True)
        if loss:
            return 0, 0, 0, 0, 0,\
                   sg_recon/b, sg_ade_min, sg_ade_avg, sg_ade_std
        else:
            return 0, 0, 0, \
                   sg_ade_min, sg_ade_avg, sg_ade_std


    def check_feat(self, data_loader):
        self.set_mode(train=False)

        with torch.no_grad():
            b = 0
            for batch in data_loader:
                b += 1
                (obs_traj, fut_traj, obs_traj_st, fut_traj_st, seq_start_end,
                 videos, classes, global_map, homo,
                 local_map, local_ic, local_homo) = batch

                idx=398
                batch = data_loader.dataset.__getitem__(idx)
                (obs_traj, fut_traj,
                 videos, classes, global_map, homo,
                 local_map, local_ic, local_homo, scale) = batch
                local_map = np.expand_dims(local_map, 0)

                obs_heat_map, sg_heat_map, lg_heat_map = self.make_heatmap(local_ic, local_map)

                self.lg_cvae.forward(obs_heat_map, None, training=False)
                recon_lg_heat = self.lg_cvae.forward(obs_heat_map, lg_heat_map, training=True)

                ###################################################
                i = 0
                plt.imshow(local_map[0])

                # ----------- 12 traj
                # heat_map_traj = np.zeros((160, 160))
                heat_map_traj = local_map[0]
                # for t in range(self.obs_len):
                # for t in [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 19]:
                for t in range(60):
                    heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 100
                    # as Y-net used variance 4 for the GT heatmap representation.
                heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
                plt.imshow(heat_map_traj)
                # plt.imshow(np.stack([heat_map_traj, local_map[i,0]],axis=2))


                # ----------- feature map
                fig = plt.figure(figsize=(5, 5))
                k = 0
                for m in self.lg_cvae.unet_features[i]:
                    k += 1
                    ax = fig.add_subplot(4, 8, k)
                    ax.imshow(m)
                    ax.axis('off')

                ###################################################
                # -------- long term goal --------

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


                #------- plot -----------
                env = local_map[0]
                # env = cv2.resize(env, (256,256))
                # for t in [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 19]:
                #     env[local_ic[i, t, 0], local_ic[i, t, 1]] = 0

                heat_map_traj = np.zeros_like(env)
                for t in [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 19]:
                    heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 50
                heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=8)
                heat_map_traj = 1-heat_map_traj / heat_map_traj.max()

                # plt.imshow(np.stack([heat_map_traj, heat_map_traj, heat_map_traj], axis=2))

                all_pred = []
                for k in range(5):
                    a = mm[k][i, 0]
                    a = nnf.interpolate(torch.tensor(a).unsqueeze(0).unsqueeze(0),
                                        size=local_map[0].shape, mode='bicubic',
                                        align_corners=False).squeeze(0).squeeze(0).detach().cpu().numpy().copy()
                    all_pred.append(1 - a / a.max())
                for k in range(5):
                    a = mmm[k][i, 0].detach().cpu().numpy().copy()
                    a = nnf.interpolate(torch.tensor(a).unsqueeze(0).unsqueeze(0),
                                        size=local_map[0].shape, mode='bicubic',
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
                        ax.imshow(np.stack([heat_map_traj, all_pred[k], 1 - env/5], axis=2))
                    else:
                        ax.imshow(mm[k % 5][i, 0])

                for k in range(10):
                    ax = fig.add_subplot(4, 5, k + 11)
                    ax.set_title('prior' + str(k % 5 + 6))
                    if k < 5:
                        ax.imshow(np.stack([heat_map_traj, all_pred[k+5], 1 - env/5], axis=2))
                    else:
                        ax.imshow(mmm[k % 5][i, 0])

                plt.imshow(env)
                tmp_idx = i


                ####################### SG ############################

                pred_lg_heat = mm[2]
                pred_lg_ics = []
                pred_lg_wc = []
                for i in range(len(obs_heat_map)):
                    map_size = local_map[0].shape
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
                        torch.transpose(torch.from_numpy(local_homo).float(), 1, 0))
                    pred_lg_wc.append(back_wc[0, :2] / back_wc[0, 2])
                    # ((back_wc - fut_traj[[3, 7, 11], 0, :2]) ** 2).sum(1).mean()
                pred_lg_wc = torch.stack(pred_lg_wc)

                # -------- short term goal --------
                pred_lg_heat_from_ic = []
                for i in range(len(pred_lg_ics)):
                    pred_lg_heat_from_ic.append(self.make_one_heatmap(local_map[0], pred_lg_ics[i][
                        0].detach().cpu().numpy().astype(int)))
                pred_lg_heat_from_ic = torch.tensor(np.stack(pred_lg_heat_from_ic)).unsqueeze(1).float().to(
                    self.device)
                pred_sg = F.sigmoid(
                    self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat_from_ic], dim=1)))

                # env = local_map[i][0]
                # heat_map_traj = np.zeros_like(env)
                # for t in [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 19]:
                #     heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 50
                # heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=8)
                # heat_map_traj = 1-heat_map_traj / heat_map_traj.max()

                i =tmp_idx
                m = pred_sg.detach().cpu().numpy().copy()[i]
                m = nnf.interpolate(torch.tensor(m).unsqueeze(0),
                                    size=local_map[0].shape, mode='bicubic',
                                    align_corners=False).squeeze(0).detach().cpu().numpy().copy()

                fig = plt.figure(figsize=(10, 5))

                for k in range(len(m)):
                    ax = fig.add_subplot(2,5,k+1)
                    ax.set_title('sg' + str(k+1))
                    ax.imshow(m[k])



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
                (obs_traj, fut_traj, seq_start_end,
                 obs_frames, pred_frames, map_path, inv_h_t,
                 local_map, local_ic, local_homo) = batch
                batch_size = obs_traj.size(1)
                total_traj += fut_traj.size(1)

                obs_heat_map, _ = self.make_heatmap(local_ic, local_map)

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
                        pred_lg_ic = []
                        for heat_map in pred_lg_heat[i]:
                            pred_lg_ic.append((heat_map == torch.max(heat_map)).nonzero()[0])
                        pred_lg_ic = torch.stack(pred_lg_ic).float()
                        pred_lg_ics.append(pred_lg_ic)

                        # ((local_ic[0,[11,15,19]] - pred_sg_ic) ** 2).sum(1).mean()
                        back_wc = torch.matmul(
                            torch.cat([pred_lg_ic, torch.ones((len(pred_lg_ic), 1)).to(self.device)], dim=1),
                            torch.transpose(local_homo[i], 1, 0))
                        pred_lg_wc.append(back_wc[0,:2] / back_wc[0,2])
                        # ((back_wc - fut_traj[[3, 7, 11], 0, :2]) ** 2).sum(1).mean()
                    pred_lg_wc = torch.stack(pred_lg_wc)
                    pred_lg_wcs.append(pred_lg_wc)

                    # -------- short term goal --------
                    # obs_lg_heat = torch.cat([obs_heat_map, pred_lg_heat[:, -1].unsqueeze(1)], dim=1)

                    if generate_heat:
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
                (hx, mux, log_varx) \
                    = self.encoderMx(obs_traj, seq_start_end, self.lg_cvae.unet_enc_feat, local_homo)

                p_dist = Normal(mux, torch.sqrt(torch.exp(log_varx)))
                z_priors = []
                for _ in range(traj_num):
                    z_priors.append(p_dist.sample())

                for pred_sg_wc in pred_sg_wcs:
                    for z_prior in z_priors:
                        # -------- trajectories --------
                        # NO TF, pred_goals, z~prior
                        fut_rel_pos_dist_prior = self.decoderMy(
                            obs_traj[-1],
                            hx,
                            z_prior,
                            pred_sg_wc,  # goal
                            self.sg_idx-3
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
                                           - fut_traj[self.sg_idx,:,:2].unsqueeze(0).repeat((lg_num,1,1,1)))**2).sum(-1)).sum(1)) # 20, 3, 4, 2
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





    ####
    def viz_init(self):
        self.viz.close(env=self.name + '/lines', win=self.win_id['test_total_loss'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['lg_fde_min'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['lg_fde_avg'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['lg_fde_std'])

        self.viz.close(env=self.name + '/lines', win=self.win_id['sg_recon'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['test_sg_recon'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['sg_ade_min'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['sg_ade_avg'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['sg_ade_std'])

    ####
    def visualize_line(self):

        # prepare data to plot
        data = self.line_gather.data
        iters = torch.Tensor(data['iter'])
        test_total_loss = torch.Tensor(data['test_total_loss'])

        sg_ade_min = torch.Tensor(data['sg_ade_min'])
        sg_ade_avg = torch.Tensor(data['sg_ade_avg'])
        sg_ade_std = torch.Tensor(data['sg_ade_std'])
        sg_recon = torch.Tensor(data['sg_recon'])
        test_sg_recon = torch.Tensor(data['test_sg_recon'])

        lg_fde_min = torch.Tensor(data['lg_fde_min'])
        lg_fde_avg = torch.Tensor(data['lg_fde_avg'])
        lg_fde_std = torch.Tensor(data['lg_fde_std'])


        self.viz.line(
            X=iters, Y=lg_fde_min, env=self.name + '/lines',
            win=self.win_id['lg_fde_min'], update='append',
            opts=dict(xlabel='iter', ylabel='lg_fde_min',
                      title='lg_fde_min'),
        )
        self.viz.line(
            X=iters, Y=lg_fde_avg, env=self.name + '/lines',
            win=self.win_id['lg_fde_avg'], update='append',
            opts=dict(xlabel='iter', ylabel='lg_fde_avg',
                      title='lg_fde_avg'),
        )
        self.viz.line(
            X=iters, Y=lg_fde_std, env=self.name + '/lines',
            win=self.win_id['lg_fde_std'], update='append',
            opts=dict(xlabel='iter', ylabel='lg_fde_std',
                      title='lg_fde_std'),
        )

        self.viz.line(
            X=iters, Y=sg_ade_std, env=self.name + '/lines',
            win=self.win_id['sg_ade_std'], update='append',
            opts=dict(xlabel='iter', ylabel='sg_ade_std',
                      title='sg_ade_std')
        )

        self.viz.line(
            X=iters, Y=sg_ade_avg, env=self.name + '/lines',
            win=self.win_id['sg_ade_avg'], update='append',
            opts=dict(xlabel='iter', ylabel='sg_ade_avg',
                      title='sg_ade_avg')
        )

        self.viz.line(
            X=iters, Y=sg_ade_min, env=self.name + '/lines',
            win=self.win_id['sg_ade_min'], update='append',
            opts=dict(xlabel='iter', ylabel='sg_ade_min',
                      title='sg_ade_min')
        )

        self.viz.line(
            X=iters, Y=sg_recon, env=self.name + '/lines',
            win=self.win_id['sg_recon'], update='append',
            opts=dict(xlabel='iter', ylabel='sg_recon',
                      title='sg_recon')
        )

        self.viz.line(
            X=iters, Y=test_sg_recon, env=self.name + '/lines',
            win=self.win_id['test_sg_recon'], update='append',
            opts=dict(xlabel='iter', ylabel='test_sg_recon',
                      title='test_sg_recon')
        )


        self.viz.line(
            X=iters, Y=test_total_loss, env=self.name + '/lines',
            win=self.win_id['test_total_loss'], update='append',
            opts=dict(xlabel='iter', ylabel='elbo',
                      title='test_elbo')
        )



    def set_mode(self, train=True):

        if train:
            self.sg_unet.train()
        else:
            self.sg_unet.eval()

    ####
    def save_checkpoint(self, iteration):
        sg_unet_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_sg_unet.pt' % iteration
        )
        mkdirs(self.ckpt_dir)
        torch.save(self.sg_unet, sg_unet_path)


    ####
    def load_checkpoint(self):

        sg_unet_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_sg_unet.pt' % self.ckpt_load_iter
        )


        if self.device == 'cuda':
            # self.encoderMx = torch.load(encoderMx_path)
            # self.encoderMy = torch.load(encoderMy_path)
            # self.decoderMy = torch.load(decoderMy_path)
            # self.lg_cvae = torch.load(lg_cvae_path)
            self.sg_unet = torch.load(sg_unet_path)
        else:
            # self.encoderMx = torch.load(encoderMx_path, map_location='cpu')
            # self.encoderMy = torch.load(encoderMy_path, map_location='cpu')
            # self.decoderMy = torch.load(decoderMy_path, map_location='cpu')
            # self.lg_cvae = torch.load(lg_cvae_path, map_location='cpu')
            # sg_unet_path = 'ckpts/ki.sg_lr_0.001_a_0.25_r_2.0_aug_1_num_sg_10_run_0/iter_5400_sg_unet.pt'
            sg_unet_path = 'ckpts/ki.sg_lr_0.001_a_0.25_r_2.0_aug_1_num_sg_2_run_1/iter_1080_sg_unet.pt'
            self.sg_unet = torch.load(sg_unet_path, map_location='cpu')
         ####

    def pretrain_load_checkpoint(self, traj, lg):

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
        else:
            self.encoderMx = torch.load(encoderMx_path, map_location='cpu')
            self.encoderMy = torch.load(encoderMy_path, map_location='cpu')
            self.decoderMy = torch.load(decoderMy_path, map_location='cpu')
            self.lg_cvae = torch.load(lg_cvae_path, map_location='cpu')