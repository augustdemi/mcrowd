import os

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
from data.nuscenes.config import Config
from data.nuscenes_dataloader import data_generator
import numpy as np
import visdom
import cv2

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
        args.num_sg = args.load_e
        self.name = '%s_zD_%s_dr_mlp_%s_dr_rnn_%s_enc_hD_%s_dec_hD_%s_mlpD_%s_map_featD_%s_map_mlpD_%s_lr_%s_klw_%s_ll_prior_w_%s_zfb_%s_scale_%s_num_sg_%s' % \
                    (args.dataset_name, args.zS_dim, args.dropout_mlp, args.dropout_rnn, args.encoder_h_dim,
                     args.decoder_h_dim, args.mlp_dim, args.map_feat_dim , args.map_mlp_dim, args.lr_VAE, args.kl_weight, args.ll_prior_w, args.fb, args.scale, args.num_sg)


        # to be appended by run_id

        self.device = args.device
        self.dt=0.1
        self.eps=1e-9
        self.ll_prior_w =args.ll_prior_w
        self.sg_idx = np.array(range(40))
        num_sg = args.load_e
        self.sg_idx = np.flip(39-self.sg_idx[::(40//num_sg)])
        print('>>>>> sg location: ', self.sg_idx + 1)

        self.z_fb = args.fb
        self.scale = args.scale

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
                recon='win_recon', loss_kl='win_loss_kl', loss_recon='win_loss_recon',
                ade_min='win_ade_min', fde_min='win_fde_min', ade_avg='win_ade_avg', fde_avg='win_fde_avg',
                ade_std='win_ade_std', fde_std='win_fde_std',
                test_loss_recon='win_test_loss_recon', test_loss_kl='win_test_loss_kl',
                loss_recon_prior='win_loss_recon_prior',
            )
            self.line_gather = DataGather(
                'iter', 'loss_recon', 'loss_kl',  'loss_recon_prior',
                'ade_min', 'fde_min', 'ade_avg', 'fde_avg', 'ade_std', 'fde_std',
                'test_loss_recon', 'test_loss_kl'
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

        # outputs
        self.output_dir_recon = os.path.join("outputs", self.name + '_recon')
        # dir for reconstructed images
        self.output_dir_synth = os.path.join("outputs", self.name + '_synth')
        # dir for synthesized images
        self.output_dir_trvsl = os.path.join("outputs", self.name + '_trvsl')

        #### create a new model or load a previously saved model

        self.ckpt_load_iter = args.ckpt_load_iter

        self.obs_len = 20
        self.pred_len = 40
        self.num_layers = args.num_layers
        self.decoder_h_dim = args.decoder_h_dim

        if self.ckpt_load_iter == 0 or args.dataset_name =='all':  # create a new model

            lg_cvae_path = 'ki.lgcvae_enc_block_1_fcomb_block_2_wD_10_lr_0.0001_lg_klw_1.0_a_0.25_r_2.0_fb_2.5_anneal_e_10_aug_1_llprior_1.0_run_0'
            lg_cvae_path = os.path.join('ckpts', lg_cvae_path, 'iter_20250_lg_cvae.pt')

            if self.device == 'cuda':
                self.lg_cvae = torch.load(lg_cvae_path)
            else:
                self.lg_cvae = torch.load(lg_cvae_path, map_location='cpu')
            print(">>>>>>>>> Init: ", lg_cvae_path)



            self.encoderMx = EncoderX(
                args.zS_dim,
                enc_h_dim=args.encoder_h_dim,
                mlp_dim=args.mlp_dim,
                map_mlp_dim=args.map_mlp_dim,
                map_feat_dim=args.map_feat_dim,
                num_layers=args.num_layers,
                dropout_mlp=args.dropout_mlp,
                dropout_rnn=args.dropout_rnn,
                device=self.device).to(self.device)
            self.encoderMy = EncoderY(
                args.zS_dim,
                enc_h_dim=args.encoder_h_dim,
                mlp_dim=args.mlp_dim,
                num_layers=args.num_layers,
                dropout_mlp=args.dropout_mlp,
                dropout_rnn=args.dropout_rnn,
                device=self.device).to(self.device)
            self.decoderMy = Decoder(
                self.pred_len,
                dec_h_dim=self.decoder_h_dim,
                enc_h_dim=args.encoder_h_dim,
                mlp_dim=args.mlp_dim,
                z_dim=args.zS_dim,
                num_layers=args.num_layers,
                device=args.device,
                dropout_rnn=args.dropout_rnn,
                scale=args.scale,
                dt=self.dt).to(self.device)

        else:  # load a previously saved model
            print('Loading saved models (iter: %d)...' % self.ckpt_load_iter)
            self.load_checkpoint()
            print('...done')


        # get VAE parameters
        vae_params = \
            list(self.encoderMx.parameters()) + \
            list(self.encoderMy.parameters()) + \
            list(self.decoderMy.parameters())
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




    def make_heatmap(self, local_ic, local_map, aug=False, only_obs=False):
        heat_maps = []
        down_size = 256
        for i in range(len(local_ic)):
            '''
            plt.imshow(local_map[i])
            plt.scatter(local_ic[i,:4,1], local_ic[i,:4,0], s=1, c='b')
            plt.scatter(local_ic[i,4:,1], local_ic[i,4:,0], s=1, c='g')
            '''
            map_size = local_map[i].shape[0]
            env = cv2.resize(local_map[i], dsize=(down_size, down_size))
            ohm = [env / 3]
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
            if not only_obs:
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
        if not only_obs:
            return heat_maps[:,:2], heat_maps[:,2:]
        else:
            return heat_maps


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
            (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
             videos, classes, global_map, homo,
             local_map, local_ic, local_homo) = next(iterator)
            batch_size = obs_traj.size(1) #=sum(seq_start_end[:,1] - seq_start_end[:,0])

            obs_heat_map =  self.make_heatmap(local_ic, local_map, aug=False, only_obs=True)

            #-------- map encoding from lgvae --------
            unet_enc_feat = self.lg_cvae.unet.down_forward(obs_heat_map)

            #-------- trajectories --------
            (hx, mux, log_varx) \
                = self.encoderMx(obs_traj_st, seq_start_end, unet_enc_feat, local_homo, train=True)


            (muy, log_vary) \
                = self.encoderMy(obs_traj_st[-1], fut_vel_st, seq_start_end, hx, train=True)

            p_dist = Normal(mux, torch.sqrt(torch.exp(log_varx)))
            q_dist = Normal(muy, torch.sqrt(torch.exp(log_vary)))


            # TF, goals, z~posterior
            fut_rel_pos_dist_tf_post = self.decoderMy(
                obs_traj_st[-1],
                obs_traj[-1, :, :2],
                hx,
                q_dist.rsample(),
                fut_traj[list(self.sg_idx), :, :2].permute(1,0,2), # goal
                self.sg_idx,
                fut_vel_st # TF
            )


            # NO TF, predicted goals, z~prior
            fut_rel_pos_dist_prior = self.decoderMy(
                obs_traj_st[-1],
                obs_traj[-1, :, :2],
                hx,
                p_dist.rsample(),
                fut_traj[list(self.sg_idx), :, :2].permute(1, 0, 2),  # goal
                self.sg_idx,
            )


            ll_tf_post = fut_rel_pos_dist_tf_post.log_prob(fut_vel_st).sum().div(batch_size)
            ll_prior = fut_rel_pos_dist_prior.log_prob(fut_vel_st).sum().div(batch_size)

            loss_kl = kl_divergence(q_dist, p_dist)
            loss_kl = torch.clamp(loss_kl, min=self.z_fb).sum().div(batch_size)
            # print('log_likelihood:', loglikelihood.item(), ' kl:', loss_kl.item())

            loglikelihood= ll_tf_post + self.ll_prior_w * ll_prior
            traj_elbo = loglikelihood - self.kl_weight * loss_kl

            loss = - traj_elbo

            self.optim_vae.zero_grad()
            loss.backward()
            self.optim_vae.step()


            # save model parameters
            if iteration % self.ckpt_save_iter == 0:
                    self.save_checkpoint(iteration)

            # (visdom) insert current line stats
            if iteration > 2700 or iteration == 1350:
                if self.viz_on and (iteration % self.viz_ll_iter == 0):
                    ade_min, fde_min, \
                    ade_avg, fde_avg, \
                    ade_std, fde_std, \
                    test_loss_recon, test_loss_kl, = self.evaluate_dist(self.val_loader, loss=True)
                    self.line_gather.insert(iter=iteration,
                                            ade_min=ade_min,
                                            fde_min=fde_min,
                                            ade_avg=ade_avg,
                                            fde_avg=fde_avg,
                                            ade_std=ade_std,
                                            fde_std=fde_std,
                                            loss_recon=-ll_tf_post.item(),
                                            loss_recon_prior=-ll_prior.item(),
                                            loss_kl=loss_kl.item(),
                                            test_loss_recon=test_loss_recon.item(),
                                            test_loss_kl=test_loss_kl.item(),

                                            )
                    prn_str = ('[iter_%d (epoch_%d)] vae_loss: %.3f ' + \
                                  '(recon: %.3f, kl: %.3f)\n' + \
                                  'ADE min: %.2f, FDE min: %.2f, ADE avg: %.2f, FDE avg: %.2f\n'
                              ) % \
                              (iteration, epoch,
                               loss.item(), -loglikelihood.item(), loss_kl.item(),
                               ade_min, fde_min, ade_avg, fde_avg
                               )

                    print(prn_str)

                # (visdom) visualize line stats (then flush out)
                if self.viz_on and (iteration % self.viz_la_iter == 0):
                    self.visualize_line()
                    self.line_gather.flush()


    def evaluate_dist(self, data_loader, loss=False):
        self.set_mode(train=False)
        total_traj = 0

        loss_recon = loss_kl = 0


        all_ade =[]
        all_fde =[]

        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
                 videos, classes, global_map, homo,
                 local_map, local_ic, local_homo) = batch
                batch_size = obs_traj.size(1)
                total_traj += fut_traj.size(1)

                obs_heat_map = self.make_heatmap(local_ic, local_map, aug=False, only_obs=True)

                # -------- map encoding from lgvae --------
                unet_enc_feat = self.lg_cvae.unet.down_forward(obs_heat_map)

                # -------- trajectories --------
                (hx, mux, log_varx) \
                    = self.encoderMx(obs_traj_st, seq_start_end, unet_enc_feat, local_homo)
                p_dist = Normal(mux, torch.sqrt(torch.exp(log_varx)))

                fut_rel_pos_dist20 = []
                for _ in range(5):
                    # NO TF, pred_goals, z~prior
                    fut_rel_pos_dist_prior = self.decoderMy(
                        obs_traj_st[-1],
                        obs_traj[-1,:,:2],
                        hx,
                        p_dist.rsample(),
                        fut_traj[list(self.sg_idx), :, :2].permute(1, 0, 2),  # goal
                        self.sg_idx,
                    )
                    fut_rel_pos_dist20.append(fut_rel_pos_dist_prior)

                if loss:

                    (muy, log_vary) \
                        = self.encoderMy(obs_traj_st[-1], fut_vel_st, seq_start_end, hx, train=False)
                    q_dist = Normal(muy, torch.sqrt(torch.exp(log_vary)))

                    loss_recon -= fut_rel_pos_dist_prior.log_prob(fut_vel_st).sum().div(batch_size)
                    kld = kl_divergence(q_dist, p_dist).sum().div(batch_size)
                    loss_kl += kld

                ade, fde = [], []
                for dist in fut_rel_pos_dist20:
                    pred_fut_traj=integrate_samples(dist.rsample() * self.scale, obs_traj[-1, :, :2], dt=self.dt)
                    ade.append(displacement_error(
                        pred_fut_traj, fut_traj[:,:,:2], mode='raw'
                    ))
                    fde.append(final_displacement_error(
                        pred_fut_traj[-1], fut_traj[-1,:,:2], mode='raw'
                    ))
                all_ade.append(torch.stack(ade))
                all_fde.append(torch.stack(fde))

            all_ade=torch.cat(all_ade, dim=1).cpu().numpy()
            all_fde=torch.cat(all_fde, dim=1).cpu().numpy()

            ade_min = np.min(all_ade, axis=0).mean()/self.pred_len
            fde_min = np.min(all_fde, axis=0).mean()
            ade_avg = np.mean(all_ade, axis=0).mean()/self.pred_len
            fde_avg = np.mean(all_fde, axis=0).mean()
            ade_std = np.std(all_ade, axis=0).mean()/self.pred_len
            fde_std = np.std(all_fde, axis=0).mean()


        self.set_mode(train=True)
        if loss:
            return ade_min, fde_min, \
                   ade_avg, fde_avg, \
                   ade_std, fde_std, \
                   loss_recon/b, loss_kl/b,
        else:
            return ade_min, fde_min, \
                   ade_avg, fde_avg, \
                   ade_std, fde_std,

    def check_feat(self, data_loader):
        self.set_mode(train=False)

        with torch.no_grad():
            b = 0
            for batch in data_loader:
                b += 1
                (obs_traj, fut_traj, seq_start_end,
                 obs_frames, pred_frames, map_path, inv_h_t,
                 local_map, local_ic, local_homo) = batch

                obs_heat_map, fut_heat_map = self.make_heatmap(local_ic, local_map)
                lg_heat_map = torch.tensor(fut_heat_map[:, 11]).float().to(self.device).unsqueeze(1)
                sg_heat_map = torch.tensor(fut_heat_map[:, self.sg_idx]).float().to(self.device)

                self.lg_cvae.forward(obs_heat_map, None, training=False)

                ###################################################
                i = 0
                plt.imshow(local_map[i, 0])

                # ----------- 12 traj
                # heat_map_traj = np.zeros((160, 160))
                heat_map_traj = local_map[i, 0].detach().cpu().numpy().copy()
                # for t in range(self.obs_len):
                for t in [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 19]:
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
                # ---------- prior
                z_prior = self.lg_cvae.prior_latent_space.rsample()
                pred_lg_prior = F.sigmoid(self.lg_cvae.fcomb.forward(self.lg_cvae.unet_features, z_prior))
                # -----------all zeros
                z = torch.zeros_like(z_prior)
                pred_lg_zeros = F.sigmoid(self.lg_cvae.fcomb.forward(self.lg_cvae.unet_features, z))
                # ---------- min/max
                z[:, :32] = 2
                z[:, 32:] = -2
                pred_lg_mm = F.sigmoid(self.lg_cvae.fcomb.forward(self.lg_cvae.unet_features, z))
                # ---------- posterior
                posterior_latent_space = self.lg_cvae.posterior.forward(obs_heat_map, lg_heat_map)
                z_post = posterior_latent_space.rsample()
                pred_lg_post = F.sigmoid(self.lg_cvae.fcomb.forward(self.lg_cvae.unet_features, z_post))
                # ---------- without latetn, only feature map
                pred_lg_path = self.lg_cvae.fcomb.last_layer(self.lg_cvae.unet_features, False)

                ###### =============== plot LG ==================#######
                fig = plt.figure(figsize=(8, 8))
                k = 0
                title = ['prior', 'post', '0', 'min/max']

                env = local_map[i,0].detach().cpu().numpy()
                heat_map_traj = np.zeros_like(env)
                for t in [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 19]:
                    heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 20
                heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)

                for m in [pred_lg_prior, pred_lg_post, pred_lg_zeros, pred_lg_mm]:
                    ax = fig.add_subplot(2, 2, k + 1)
                    ax.set_title(title[k])
                    ax.imshow(m[i, 0])
                    # ax.imshow(np.stack([m[i, 0] / m[i, 0].max(), env, heat_map_traj],axis=2))
                    k += 1


                ###################################################
                # ---------- SG
                pred_sg_heat = F.sigmoid(
                    self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_prior], dim=1), training=False))

                pred_sg_gt = F.sigmoid(
                    self.sg_unet.forward(torch.cat([obs_heat_map, lg_heat_map], dim=1), training=False))

                ###### =============== plot SG ==================#######
                fig = plt.figure(figsize=(10, 5))
                k = 0
                env = local_map[i,0].detach().cpu().numpy()

                heat_map_traj = np.zeros_like(env)
                for t in [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 19]:
                    heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 20
                heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)

                for m in pred_sg_gt[i]:
                    k += 1
                    ax = fig.add_subplot(1, 3, k)
                    ax.set_title('sg' + str(k))
                    ax.imshow(np.stack([m / m.max(), env, heat_map_traj],axis=2))



                ###################################################
                # ----------- LG ic & wc
                pred_lg_ic = []
                for heat_map in pred_lg_prior[i]:
                    pred_lg_ic.append((heat_map == torch.max(heat_map)).nonzero()[0])
                pred_lg_ic = torch.stack(pred_lg_ic).float()

                # ((local_ic[0,[11,15,19]] - pred_sg_ic) ** 2).sum(1).mean()
                back_wc = torch.matmul(
                    torch.cat([pred_lg_ic, torch.ones((len(pred_lg_ic), 1)).to(self.device)], dim=1),
                    torch.transpose(local_homo[i], 1, 0))
                pred_lg_wc = back_wc[0, :2] / back_wc[0, 2]
                # ((back_wc - fut_traj[[3, 7, 11], 0, :2]) ** 2).sum(1).mean()


                # ----------- SG ic & wc
                pred_sg_ic = []
                for heat_map in pred_sg_heat[i]:
                    pred_sg_ic.append((heat_map == torch.max(heat_map)).nonzero()[0])
                pred_sg_ic = torch.stack(pred_sg_ic).float()

                # ((local_ic[0,[11,15,19]] - pred_sg_ic) ** 2).sum(1).mean()
                back_wc = torch.matmul(
                    torch.cat([pred_sg_ic, torch.ones((len(pred_sg_ic), 1)).to(self.device)], dim=1),
                    torch.transpose(local_homo[i], 1, 0))
                back_wc /= back_wc[:, 2].unsqueeze(1)
                pred_sg_wc = back_wc[:, :2]
                # ((back_wc - fut_traj[[3, 7, 11], 0, :2]) ** 2).sum(1).mean()





    def evaluate_dist_gt_goal(self, data_loader):
        self.set_mode(train=False)
        total_traj = 0

        all_ade =[]
        all_fde =[]
        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, seq_start_end,
                 obs_frames, pred_frames, map_path, inv_h_t,
                 local_map, local_ic, local_homo) = batch
                total_traj += fut_traj.size(1)

                obs_heat_map, fut_heat_map = self.make_heatmap(local_ic, local_map)
                lg_heat_map = torch.tensor(fut_heat_map[:, 1]).float().to(self.device).unsqueeze(1)

                self.lg_cvae.forward(obs_heat_map, None, training=False)

                # -------- trajectories --------
                self.sg_unet.forward(torch.cat([obs_heat_map, lg_heat_map], dim=1), training=False)

                (hx, mux, log_varx) \
                    = self.encoderMx(obs_traj, seq_start_end, self.sg_unet.enc_feat, local_homo)
                p_dist = Normal(mux, torch.sqrt(torch.exp(log_varx)))


                # TF, goals, z~posterior


                ade, fde = [], []
                for _ in range(20):
                    fut_rel_pos_dist_prior = self.decoderMy(
                        obs_traj[-1],
                        hx,
                        p_dist.rsample(),
                        fut_traj[list(self.sg_idx), :, :2].permute(1, 0, 2),  # goal
                    )

                    pred_fut_traj = integrate_samples(fut_rel_pos_dist_prior.rsample(), obs_traj[-1, :, :2], dt=self.dt)
                    ade.append(displacement_error(
                        pred_fut_traj, fut_traj[:,:,:2], mode='raw'
                    ))
                    fde.append(final_displacement_error(
                        pred_fut_traj[-1], fut_traj[-1,:,:2], mode='raw'
                    ))

                all_ade.append(torch.stack(ade))
                all_fde.append(torch.stack(fde))


            all_ade=torch.cat(all_ade, dim=1).cpu().numpy()
            all_fde=torch.cat(all_fde, dim=1).cpu().numpy()

            ade_min = np.min(all_ade, axis=0).mean()/self.pred_len
            fde_min = np.min(all_fde, axis=0).mean()
            ade_avg = np.mean(all_ade, axis=0).mean()/self.pred_len
            fde_avg = np.mean(all_fde, axis=0).mean()
            ade_std = np.std(all_ade, axis=0).mean()/self.pred_len
            fde_std = np.std(all_fde, axis=0).mean()

            print('ade min: ', ade_min)
            print('ade avg: ', ade_avg)
            print('ade std: ', ade_std)
            print('fde min: ', fde_min)
            print('fde avg: ', fde_avg)
            print('fde std: ', fde_std)


    ####
    def viz_init(self):
        self.viz.close(env=self.name + '/lines', win=self.win_id['loss_recon'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['loss_recon_prior'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['loss_kl'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['test_loss_recon'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['test_loss_kl'])

        self.viz.close(env=self.name + '/lines', win=self.win_id['ade_min'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['fde_min'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['ade_avg'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['fde_avg'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['ade_std'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['fde_std'])
    ####
    def visualize_line(self):

        # prepare data to plot
        data = self.line_gather.data
        iters = torch.Tensor(data['iter'])
        loss_recon = torch.Tensor(data['loss_recon'])
        loss_recon_prior = torch.Tensor(data['loss_recon_prior'])
        loss_kl = torch.Tensor(data['loss_kl'])
        ade_min = torch.Tensor(data['ade_min'])
        fde_min = torch.Tensor(data['fde_min'])
        ade_avg = torch.Tensor(data['ade_avg'])
        fde_avg = torch.Tensor(data['fde_avg'])
        ade_std = torch.Tensor(data['ade_std'])
        fde_std = torch.Tensor(data['fde_std'])
        test_loss_recon = torch.Tensor(data['test_loss_recon'])
        test_loss_kl = torch.Tensor(data['test_loss_kl'])





        self.viz.line(
            X=iters, Y=loss_recon, env=self.name + '/lines',
            win=self.win_id['loss_recon'], update='append',
            opts=dict(xlabel='iter', ylabel='-loglikelihood',
                      title='Recon. loss of predicted future traj')
        )

        self.viz.line(
            X=iters, Y=loss_recon_prior, env=self.name + '/lines',
            win=self.win_id['loss_recon_prior'], update='append',
            opts=dict(xlabel='iter', ylabel='-loglikelihood',
                      title='Recon. loss - prior')
        )


        self.viz.line(
            X=iters, Y=loss_kl, env=self.name + '/lines',
            win=self.win_id['loss_kl'], update='append',
            opts=dict(xlabel='iter', ylabel='kl divergence',
                      title='KL div. btw posterior and c. prior'),
        )


        self.viz.line(
            X=iters, Y=test_loss_recon, env=self.name + '/lines',
            win=self.win_id['test_loss_recon'], update='append',
            opts=dict(xlabel='iter', ylabel='-loglikelihood',
                      title='Test Recon. loss of predicted future traj')
        )

        self.viz.line(
            X=iters, Y=test_loss_kl, env=self.name + '/lines',
            win=self.win_id['test_loss_kl'], update='append',
            opts=dict(xlabel='iter', ylabel='kl divergence',
                      title='Test KL div. btw posterior and c. prior'),
        )


        self.viz.line(
            X=iters, Y=ade_min, env=self.name + '/lines',
            win=self.win_id['ade_min'], update='append',
            opts=dict(xlabel='iter', ylabel='ade',
                      title='ADE min'),
        )
        self.viz.line(
            X=iters, Y=fde_min, env=self.name + '/lines',
            win=self.win_id['fde_min'], update='append',
            opts=dict(xlabel='iter', ylabel='fde',
                      title='FDE min'),
        )
        self.viz.line(
            X=iters, Y=ade_avg, env=self.name + '/lines',
            win=self.win_id['ade_avg'], update='append',
            opts=dict(xlabel='iter', ylabel='ade',
                      title='ADE avg'),
        )

        self.viz.line(
            X=iters, Y=fde_avg, env=self.name + '/lines',
            win=self.win_id['fde_avg'], update='append',
            opts=dict(xlabel='iter', ylabel='fde',
                      title='FDE avg'),
        )
        self.viz.line(
            X=iters, Y=ade_std, env=self.name + '/lines',
            win=self.win_id['ade_std'], update='append',
            opts=dict(xlabel='iter', ylabel='ade std',
                      title='ADE std'),
        )

        self.viz.line(
            X=iters, Y=fde_std, env=self.name + '/lines',
            win=self.win_id['fde_std'], update='append',
            opts=dict(xlabel='iter', ylabel='fde std',
                      title='FDE std'),
        )


    def set_mode(self, train=True):

        if train:
            self.encoderMx.train()
            self.encoderMy.train()
            self.decoderMy.train()
        else:
            self.encoderMx.eval()
            self.encoderMy.eval()
            self.decoderMy.eval()

    ####
    def save_checkpoint(self, iteration):

        encoderMx_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderMx.pt' % iteration
        )
        encoderMy_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderMy.pt' % iteration
        )
        decoderMy_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_decoderMy.pt' % iteration
        )
        lg_cvae_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_lg_cvae.pt' % iteration
        )
        sg_unet_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_sg_unet.pt' % iteration
        )
        mkdirs(self.ckpt_dir)

        torch.save(self.encoderMx, encoderMx_path)
        torch.save(self.encoderMy, encoderMy_path)
        torch.save(self.decoderMy, decoderMy_path)
    ####
    def load_checkpoint(self):

        encoderMx_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderMx.pt' % self.ckpt_load_iter
        )
        encoderMy_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderMy.pt' % self.ckpt_load_iter
        )
        decoderMy_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_decoderMy.pt' % self.ckpt_load_iter
        )
        lg_cvae_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_lg_cvae.pt' % self.ckpt_load_iter
        )
        sg_unet_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_sg_unet.pt' % self.ckpt_load_iter
        )



        if self.device == 'cuda':
            self.encoderMx = torch.load(encoderMx_path)
            self.encoderMy = torch.load(encoderMy_path)
            self.decoderMy = torch.load(decoderMy_path)
        else:
            self.encoderMx = torch.load(encoderMx_path, map_location='cpu')
            self.encoderMy = torch.load(encoderMy_path, map_location='cpu')
            self.decoderMy = torch.load(decoderMy_path, map_location='cpu')