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
        self.name = '%s_zD_%s_dr_mlp_%s_dr_rnn_%s_enc_hD_%s_dec_hD_%s_mlpD_%s_map_featD_%s_map_mlpD_%s_lr_%s_klw_%s_ll_prior_w_%s_zfb_%s_scale_%s_num_sg_%s' \
                    'ctxtD_%s_coll_th_%s_w_agent_%s_beta_%s_lr_e_%s_w_map_%s' % \
                    (args.dataset_name, args.zS_dim, args.dropout_mlp, args.dropout_rnn, args.encoder_h_dim,
                     args.decoder_h_dim, args.mlp_dim, args.map_feat_dim , args.map_mlp_dim, args.lr_VAE, args.kl_weight,
                     args.ll_prior_w, args.fb, args.scale, args.num_sg, args.context_dim, args.coll_th, args.w_agent, args.beta, args.lr_e, args.w_map)

        # to be appended by run_id

        # self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = args.device
        self.temp=1.99
        self.dt=0.5
        self.eps=1e-9
        self.ll_prior_w =args.ll_prior_w
        self.sg_idx = np.array(range(12))
        self.sg_idx = np.flip(11-self.sg_idx[::(12//args.num_sg)])


        self.coll_th = args.coll_th
        self.beta = args.beta
        self.context_dim = args.context_dim
        self.w_agent = args.w_agent
        self.w_map = args.w_map

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
                loss_recon_prior='win_loss_recon_prior', loss_coll='win_loss_coll', test_loss_coll = 'win_test_loss_coll',
                test_total_coll='win_test_total_coll', total_coll='win_total_coll', map_coll='map_coll',
                test_map_coll='test_map_coll'
            )
            self.line_gather = DataGather(
                'iter', 'loss_recon', 'loss_kl', 'loss_recon_prior',
                'ade_min', 'fde_min', 'ade_avg', 'fde_avg', 'ade_std', 'fde_std',
                'test_loss_recon', 'test_loss_kl', 'test_loss_coll', 'loss_coll', 'test_total_coll', 'total_coll',
                'map_coll', 'test_map_coll'
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

        self.obs_len = 4
        self.pred_len = args.pred_len
        self.num_layers = args.num_layers
        self.decoder_h_dim = args.decoder_h_dim

        lg_cvae_path = 'nu.lgcvae_enc_block_1_fcomb_block_2_wD_10_lr_0.0001_lg_klw_1.0_a_0.25_r_2.0_fb_3.0_anneal_e_10_aug_1_llprior_0.0_run_4'
        lg_cvae_path = os.path.join('ckpts', lg_cvae_path, 'iter_39000_lg_cvae.pt')

        if self.device == 'cuda':
            self.lg_cvae = torch.load(lg_cvae_path)
        else:
            self.lg_cvae = torch.load(lg_cvae_path, map_location='cpu')
        print(">>>>>>>>> Init: ", lg_cvae_path)

        if self.ckpt_load_iter == 0 or args.dataset_name =='all':  # create a new model

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
                args.pred_len,
                dec_h_dim=self.decoder_h_dim,
                enc_h_dim=args.encoder_h_dim,
                map_feat_dim=args.map_feat_dim,
                mlp_dim=args.mlp_dim,
                z_dim=args.zS_dim,
                num_layers=args.num_layers,
                device=args.device,
                dropout_rnn=args.dropout_rnn,
                scale=args.scale,
                dt=self.dt,
                context_dim=args.context_dim).to(self.device)


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
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optim_vae,
                                        lr_lambda=lambda epoch: args.lr_e ** epoch)

        # prepare dataloader (iterable)
        print('Start loading data...')


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




    def temmp(self):
        # aa = torch.zeros((120, 2, 256, 256)).to(self.device)
        # self.lg_cvae.unet.down_forward(aa)
        print('t')

    ## https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e
    def bilinear_interpolate_map(self, local_map, local_homo, pred_traj):
        map_dim = local_map.shape[0]
        pred_ic = torch.matmul(torch.cat([pred_traj, torch.ones(len(pred_traj), 1).to(self.device)], 1),
                                    torch.pinverse(local_homo.transpose(1, 0)))
        pred_ic = pred_ic / torch.unsqueeze(pred_ic[:, 2], 1)
        pred_ic = pred_ic[:, :2]
        '''
        plt.imshow(local_map)
        plt.scatter(pred_ic[:,1].detach().numpy(), pred_ic[:,0].detach().numpy(), s=1, c='r')
        '''
        pred_ic[:, 0] = pred_ic[:, 0] / (map_dim - 1)  # normalize to between  0 and 1
        pred_ic[:, 1] = pred_ic[:, 1] / (map_dim - 1)  # normalize to between  0 and 1
        pred_ic = pred_ic * 2 - 1
        return torch.nn.functional.grid_sample(local_map.transpose(1,0).unsqueeze(0).unsqueeze(0).repeat((self.pred_len, 1,1,1)), pred_ic.unsqueeze(1).unsqueeze(1)).sum()

    def resize_map(self,local_map):
        resized_map = []
        for m in local_map:
            resized_map.append(cv2.resize(m, dsize=(256, 256)))
        return torch.tensor(resized_map).unsqueeze(1).to(self.device)

    ####
    def train(self):
        self.set_mode(train=True)
        data_loader = self.train_loader

        iter_per_epoch = len(data_loader.idx_list)
        start_iter = self.ckpt_load_iter + 1
        epoch = int(start_iter / iter_per_epoch)

        e_coll_loss = 0
        e_total_coll = 0

        for iteration in range(start_iter, self.max_iter + 1):
            data = data_loader.next_sample()
            if data is None:
                print(0)
                continue
            # reset data iterators for each epoch
            if iteration % iter_per_epoch == 0:
                if self.ckpt_load_iter > 0:
                    data_loader.is_epoch_end(force=True)
                else:
                    data_loader.is_epoch_end()
                print('==== epoch %d done ====' % epoch)
                epoch +=1
                if self.optim_vae.param_groups[0]['lr'] > 5e-5:
                    self.scheduler.step()
                else:
                    self.optim_vae.param_groups[0]['lr'] = 5e-5
                print("lr: ", self.optim_vae.param_groups[0]['lr'])
                print('e_coll_loss: ', e_coll_loss, ' // e_total_coll: ', e_total_coll)
                prev_e_total_coll = e_total_coll
                e_coll_loss = 0
                e_total_coll = 0

            # ============================================
            #          TRAIN THE VAE (ENC & DEC)
            # ============================================
            (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
             maps, local_map, local_ic, local_homo) = data
            batch_size = fut_traj.size(1) #=sum(seq_start_end[:,1] - seq_start_end[:,0])

            resized_map = self.resize_map(local_map)

            #-------- trajectories --------
            (hx, mux, log_varx) \
                = self.encoderMx(obs_traj_st, seq_start_end, resized_map, train=True)


            (muy, log_vary) \
                = self.encoderMy(obs_traj_st[-1], fut_vel_st, seq_start_end, resized_map, train=True)

            p_dist = Normal(mux, torch.clamp(torch.sqrt(torch.exp(log_varx)), min=1e-8))
            q_dist = Normal(muy, torch.clamp(torch.sqrt(torch.exp(log_vary)), min=1e-8))


            # TF, goals, z~posterior
            fut_rel_pos_dist_tf_post = self.decoderMy(
                seq_start_end,
                obs_traj_st[-1],
                obs_traj[-1, :, :2],
                hx,
                q_dist.rsample(),
                fut_traj[list(self.sg_idx), :, :2].permute(1,0,2), # goal
                self.sg_idx,
                fut_vel_st, # TF
                train=True
            )


            # NO TF, predicted goals, z~prior
            fut_rel_pos_dist_prior = self.decoderMy(
                seq_start_end,
                obs_traj_st[-1],
                obs_traj[-1, :, :2],
                hx,
                p_dist.rsample(),
                fut_traj[list(self.sg_idx), :, :2].permute(1, 0, 2),  # goal
                self.sg_idx,
                train=True
            )


            ll_tf_post = fut_rel_pos_dist_tf_post.log_prob(fut_vel_st).sum().div(batch_size)
            ll_prior = fut_rel_pos_dist_prior.log_prob(fut_vel_st).sum().div(batch_size)

            loss_kl = kl_divergence(q_dist, p_dist)
            loss_kl = torch.clamp(loss_kl, min=self.z_fb).sum().div(batch_size)
            # print('log_likelihood:', loglikelihood.item(), ' kl:', loss_kl.item())

            loglikelihood= ll_tf_post + self.ll_prior_w * ll_prior
            traj_elbo = loglikelihood - self.kl_weight * loss_kl



            coll_loss = torch.tensor(0.0).to(self.device)
            total_coll = 0
            n_scene = 0

            pred_fut_traj = integrate_samples(fut_rel_pos_dist_prior.rsample() * self.scale, obs_traj[-1, :, :2],
                                              dt=self.dt)

            pred_fut_traj_post = integrate_samples(fut_rel_pos_dist_tf_post.rsample() * self.scale,
                                                   obs_traj[-1, :, :2],
                                                   dt=self.dt)
            if self.w_agent > 0:
                for s, e in seq_start_end:
                    n_scene += 1
                    num_ped = e - s
                    if num_ped == 1:
                        continue
                    for t in range(self.pred_len):
                        ## prior
                        curr1 = pred_fut_traj[t, s:e].repeat(num_ped, 1)
                        curr2 = self.repeat(pred_fut_traj[t, s:e], num_ped)
                        dist = torch.norm(curr1 - curr2, dim=1)
                        dist = dist.reshape(num_ped, num_ped)
                        diff_agent_dist = dist[torch.where(dist > 0)]
                        coll_loss += (torch.sigmoid(-(diff_agent_dist - self.coll_th) * self.beta)).sum()
                        total_coll += (len(torch.where(diff_agent_dist < 1.5)[0]) / 2)
                        ## posterior
                        curr1_post = pred_fut_traj_post[t, s:e].repeat(num_ped, 1)
                        curr2_post = self.repeat(pred_fut_traj_post[t, s:e], num_ped)
                        dist_post = torch.norm(curr1_post - curr2_post, dim=1)
                        dist_post = dist_post.reshape(num_ped, num_ped)
                        diff_agent_dist_post = dist_post[torch.where(dist_post > 0)]
                        coll_loss += (torch.sigmoid(-(diff_agent_dist_post - self.coll_th) * self.beta)).sum()
                        total_coll += (len(torch.where(diff_agent_dist_post < 1.5)[0]) / 2)

            # pred_wcs = fut_traj[:,:,:2].transpose(1,0) # batch size, past step, 2
            pred_wcs = pred_fut_traj.transpose(1,0) # batch size, past step, 2
            pred_wcs_post = pred_fut_traj_post.transpose(1,0) # batch size, past step, 2
            map_coll_loss = 0
            for i in range(len(pred_wcs)):
                this_map = local_map[i].copy()
                this_map[np.where(this_map>0)]=1
                this_map = torch.tensor(this_map).to(self.device)
                map_coll_loss += self.bilinear_interpolate_map(this_map, local_homo[i], pred_wcs[i])
                map_coll_loss += self.bilinear_interpolate_map(this_map, local_homo[i], pred_wcs_post[i])

            loss = - traj_elbo + self.w_agent * coll_loss + self.w_map * map_coll_loss
            e_coll_loss +=coll_loss.item()
            e_total_coll +=total_coll

            self.optim_vae.zero_grad()
            loss.backward()
            self.optim_vae.step()



            # save model parameters
            if (iteration % (iter_per_epoch*5) == 0):
                self.save_checkpoint(iteration)

            # (visdom) insert current line stats
            if iteration > 0:
                if iteration == iter_per_epoch or (self.viz_on and (iteration % (iter_per_epoch*5) == 0)):

                    ade_min, fde_min, \
                    ade_avg, fde_avg, \
                    ade_std, fde_std, \
                    test_loss_recon, test_loss_kl, test_loss_coll, test_total_coll, test_map_coll_loss = self.evaluate_dist(self.val_loader, loss=True)
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
                                            loss_coll=coll_loss.item(),
                                            total_coll=prev_e_total_coll,
                                            map_coll=map_coll_loss.item(),
                                            test_map_coll=test_map_coll_loss.item(),
                                            test_loss_recon=test_loss_recon.item(),
                                            test_loss_kl=test_loss_kl.item(),
                                            test_loss_coll=test_loss_coll.item(),
                                            test_total_coll=test_total_coll
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
                    self.visualize_line()
                    self.line_gather.flush()



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


    def evaluate_dist(self, data_loader, loss=False):
        self.set_mode(train=False)
        total_traj = 0

        loss_recon = loss_kl = 0
        coll_loss = 0
        total_coll = 0
        n_scene = 0
        map_coll_loss = 0

        all_ade =[]
        all_fde =[]

        with torch.no_grad():
            b=0
            while not data_loader.is_epoch_end():
                data = data_loader.next_sample()
                if data is None:
                    continue
                b+=1
                (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
                 maps, local_map, local_ic, local_homo) = data
                batch_size = fut_traj.size(1)
                total_traj += fut_traj.size(1)

                resized_map= self.resize_map(local_map)

                # -------- trajectories --------
                (hx, mux, log_varx) \
                    = self.encoderMx(obs_traj_st, seq_start_end, resized_map)
                p_dist = Normal(mux, torch.clamp(torch.sqrt(torch.exp(log_varx)), min=1e-8))

                fut_rel_pos_dist20 = []
                for _ in range(4):
                    # NO TF, pred_goals, z~prior
                    fut_rel_pos_dist_prior = self.decoderMy(
                        seq_start_end,
                        obs_traj_st[-1],
                        obs_traj[-1, :, :2],
                        hx,
                        p_dist.rsample(),
                        fut_traj[list(self.sg_idx), :, :2].permute(1, 0, 2),  # goal
                        self.sg_idx,
                    )
                    fut_rel_pos_dist20.append(fut_rel_pos_dist_prior)

                if loss:
                    (muy, log_vary) \
                        = self.encoderMy(obs_traj_st[-1], fut_vel_st, seq_start_end, resized_map, train=False)
                    q_dist = Normal(muy, torch.sqrt(torch.exp(log_vary)))

                    loss_recon -= fut_rel_pos_dist_prior.log_prob(fut_vel_st).sum().div(batch_size)
                    kld = kl_divergence(q_dist, p_dist).sum().div(batch_size)
                    loss_kl += kld

                    pred_fut_traj = integrate_samples(fut_rel_pos_dist_prior.rsample() * self.scale,
                                                      obs_traj[-1, :, :2],
                                                      dt=self.dt)
                    for s, e in seq_start_end:
                        n_scene += 1
                        num_ped = e - s
                        if num_ped == 1:
                            continue
                        seq_traj = pred_fut_traj[:, s:e]
                        for i in range(len(seq_traj)):
                            curr1 = seq_traj[i].repeat(num_ped, 1)
                            curr2 = self.repeat(seq_traj[i], num_ped)
                            dist = torch.norm(curr1 - curr2, dim=1)
                            dist = dist.reshape(num_ped, num_ped)
                            diff_agent_dist = dist[torch.where(dist > 0)]
                            # diff_agent_dist[torch.where(diff_agent_dist > self.coll_th)] += self.beta
                            coll_loss += (torch.sigmoid(-(diff_agent_dist - self.coll_th) * self.beta)).sum()
                            total_coll += (len(torch.where(diff_agent_dist < 1.5)[0]) / 2)

                    pred_wcs = pred_fut_traj.transpose(1, 0)  # batch size, past step, 2
                    for i in range(len(pred_wcs)):
                        this_map = local_map[i].copy()
                        this_map[np.where(this_map > 0)] = 1
                        this_map = torch.tensor(this_map).to(self.device)
                        map_coll_loss += self.bilinear_interpolate_map(this_map, local_homo[i], pred_wcs[i])



                ade, fde = [], []
                for dist in fut_rel_pos_dist20:
                    pred_fut_traj = integrate_samples(dist.rsample() * self.scale, obs_traj[-1, :, :2], dt=self.dt)
                    ade.append(displacement_error(
                        pred_fut_traj, fut_traj[:, :, :2], mode='raw'
                    ))
                    fde.append(final_displacement_error(
                        pred_fut_traj[-1], fut_traj[-1, :, :2], mode='raw'
                    ))
                all_ade.append(torch.stack(ade))
                all_fde.append(torch.stack(fde))

            all_ade = torch.cat(all_ade, dim=1).cpu().numpy()
            all_fde = torch.cat(all_fde, dim=1).cpu().numpy()

            ade_min = np.min(all_ade, axis=0).mean() / self.pred_len
            fde_min = np.min(all_fde, axis=0).mean()
            ade_avg = np.mean(all_ade, axis=0).mean() / self.pred_len
            fde_avg = np.mean(all_fde, axis=0).mean()
            ade_std = np.std(all_ade, axis=0).mean() / self.pred_len
            fde_std = np.std(all_fde, axis=0).mean()

        self.set_mode(train=True)
        if loss:
            return ade_min, fde_min, \
                   ade_avg, fde_avg, \
                   ade_std, fde_std, \
                   loss_recon / b, loss_kl / b, coll_loss / b, total_coll, map_coll_loss/b
        else:
            return ade_min, fde_min, \
                   ade_avg, fde_avg, \
                   ade_std, fde_std


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




    def evaluate_collision(self, data_loader, num_samples, threshold):
        self.set_mode(train=False)
        total_traj = 0
        all_coll = []

        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_vel, fut_traj_vel, seq_start_end, obs_frames, fut_frames, past_obst,
                 fut_obst) = batch
                total_traj += fut_traj.size(1)


                (encX_h_feat, logitX) \
                    = self.encoderMx(obs_traj, seq_start_end)
                relaxed_p_dist = concrete(logits=logitX, temperature=self.temp)

                coll_20samples = [] # (20, # seq, 12)
                for _ in range(num_samples):
                    fut_rel_pos_dist = self.decoderMy(
                        obs_traj[-1],
                        encX_h_feat,
                        relaxed_p_dist.rsample()
                    )
                    pred_fut_traj_rel = fut_rel_pos_dist.rsample()

                    pred_fut_traj=integrate_samples(pred_fut_traj_rel, obs_traj[-1][:, :2], dt=self.dt)

                    seq_coll = [] #64
                    for idx, (start, end) in enumerate(seq_start_end):

                        start = start.item()
                        end = end.item()
                        num_ped = end - start
                        if num_ped==1:
                            continue
                        one_frame_slide = pred_fut_traj[:,start:end,:] # (pred_len, num_ped, 2)

                        frame_coll = [] #num_ped
                        for i in range(self.pred_len):
                            curr_frame = one_frame_slide[i] # frame of time=i #(num_ped,2)
                            curr1 = curr_frame.repeat(num_ped, 1)
                            curr2 = self.repeat(curr_frame, num_ped)
                            dist = torch.sqrt(torch.pow(curr1 - curr2, 2).sum(1)).cpu().numpy()
                            dist = dist.reshape(num_ped, num_ped) # all distance between all num_ped*num_ped
                            diff_agent_idx = np.triu_indices(num_ped, k=1) # only distinct distances of num_ped C 2(upper triange except for diag)
                            diff_agent_dist = dist[diff_agent_idx]
                            curr_coll_rate = (diff_agent_dist < threshold).sum()

                            frame_coll.append(curr_coll_rate)
                        seq_coll.append(frame_coll)
                    coll_20samples.append(seq_coll)

                all_coll.append(np.array(coll_20samples))

            all_coll=np.concatenate(all_coll, axis=1) #(20,70,12)
            print('all_coll: ', all_coll.shape)
            coll_rate_min=all_coll.min(axis=0).sum()
            coll_rate_avg=all_coll.mean(axis=0).sum()
            coll_rate_std=all_coll.std(axis=0).mean()

            #non-zero coll
            non_zero_coll_avg = 0
            non_zero_coll_min = 0
            non_zero_coll_std = 0

        return coll_rate_min, non_zero_coll_min, \
               coll_rate_avg, non_zero_coll_avg, \
               coll_rate_std, non_zero_coll_std




    def compute_obs_violations(self, predicted_trajs, obs_map):
        interp_obs_map = RectBivariateSpline(range(obs_map.shape[0]),
                                             range(obs_map.shape[1]),
                                             1-binary_dilation(obs_map, iterations=1),
                                             kx=1, ky=1)

        old_shape = predicted_trajs.shape
        predicted_trajs = predicted_trajs.reshape((-1,2))

        # plt.imshow(obs_map)
        # for i in range(12, 24):
        #     plt.scatter(predicted_trajs[i,0], predicted_trajs[i,1], s=1, c='r')
        #
        # a = 1-binary_dilation(obs_map, iterations=1)
        # plt.imshow(a)
        # for i in range(12):
        #     plt.scatter(predicted_trajs[i,0], predicted_trajs[i,1], s=1, c='r')

        traj_obs_values = interp_obs_map(predicted_trajs[:, 1], predicted_trajs[:, 0], grid=False)
        traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1]))
        num_viol_trajs = np.sum(traj_obs_values.max(axis=1) > 0,
                                dtype=float)  # 20개 case 각각에 대해 12 future time중 한번이라도(12개중 max) 충돌이 있었나 확인

        return num_viol_trajs, traj_obs_values.sum(axis=1)

    def map_collision(self, data_loader, num_samples=20):

        total_traj = 0
        total_viol = 0
        min_viol = []
        avg_viol = []
        std_viol = []
        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
                 obs_frames, pred_frames, obs_obst, fut_obst, map_path, inv_h_t) = batch
                total_traj += fut_traj.size(1)


                (encX_h_feat, logitX, map_featX) \
                    = self.encoderMx(obs_traj_st, seq_start_end, obs_obst, obs_traj[:,:,2:4])
                relaxed_p_dist = concrete(logits=logitX, temperature=self.temp)


                for j, (s, e) in enumerate(seq_start_end):
                    agent_rng = range(s, e)

                    multi_sample_pred = []
                    for _ in range(num_samples):
                        fut_rel_pos_dist = self.decoderMy(
                            obs_traj_st[-1],
                            encX_h_feat,
                            relaxed_p_dist.rsample(),
                            obs_obst[-1].unsqueeze(0),
                            map_info=[seq_start_end, map_path, inv_h_t,
                                      lambda x: integrate_samples(x, obs_traj[-1, :, :2], dt=self.dt)]
                        )
                        pred_fut_traj_rel = fut_rel_pos_dist.rsample()
                        pred_fut_traj = integrate_samples(pred_fut_traj_rel, obs_traj[-1, :, :2], dt=self.dt)

                        pred_data = []
                        for idx in range(len(agent_rng)):
                            one_ped = agent_rng[idx]
                            pred_real = pred_fut_traj[:, one_ped].detach().cpu().numpy()
                            pred_pixel = np.concatenate([pred_real, np.ones((self.pred_len, 1))], axis=1)
                            pred_pixel = np.matmul(pred_pixel, inv_h_t[j])
                            pred_pixel /= np.expand_dims(pred_pixel[:, 2], 1)
                            pred_data.append(pred_pixel)

                        pred_data = np.stack(pred_data)

                        multi_sample_pred.append(pred_data)

                    multi_sample_pred = np.array(multi_sample_pred)[:,:,:,:2]

                    for a in range(len(agent_rng)):
                        obs_map = imageio.imread(map_path[j])
                        num_viol_trajs, viol20 = self.compute_obs_violations(multi_sample_pred[:,a], obs_map)
                        total_viol += num_viol_trajs
                        min_viol.append(np.min(viol20))
                        avg_viol.append(np.mean(viol20))
                        std_viol.append(np.std(viol20))
        return total_viol / total_traj, np.mean(np.array(min_viol)), np.mean(np.array(avg_viol)), np.mean(np.array(std_viol))




    def evaluate_real_collision(self, data_loader, threshold):
        self.set_mode(train=False)
        total_traj = 0
        all_coll = []

        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_vel, fut_traj_vel, seq_start_end, obs_frames, fut_frames, past_obst,
                 fut_obst) = batch

                total_traj += fut_traj.size(1)

                seq_coll = []  # 64
                for idx, (start, end) in enumerate(seq_start_end):

                    start = start.item()
                    end = end.item()
                    num_ped = end - start
                    if num_ped == 1:
                        continue
                    one_frame_slide = fut_traj[:, start:end, :2]  # (pred_len, num_ped, 2)

                    frame_coll = []  # num_ped
                    for i in range(self.pred_len):
                        curr_frame = one_frame_slide[i]  # frame of time=i #(num_ped,2)
                        curr1 = curr_frame.repeat(num_ped, 1)
                        curr2 = self.repeat(curr_frame, num_ped)
                        dist = torch.sqrt(torch.pow(curr1 - curr2, 2).sum(1)).cpu().numpy()
                        dist = dist.reshape(num_ped, num_ped)  # all distance between all num_ped*num_ped
                        diff_agent_idx = np.triu_indices(num_ped,
                                                         k=1)  # only distinct distances of num_ped C 2(upper triange except for diag)
                        diff_agent_dist = dist[diff_agent_idx]
                        curr_coll_rate = (diff_agent_dist < threshold).sum()
                        frame_coll.append(curr_coll_rate)
                    seq_coll.append(frame_coll)
                all_coll.append(np.array(seq_coll))
            all_coll=np.concatenate(all_coll, axis=0) #(70,12)
            print('all_coll: ', all_coll.shape)
            coll_rate=all_coll.sum()

        self.set_mode(train=True)
        return coll_rate


    def plot_traj(self, data_loader, num_samples=20):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        gif_path = "D:\crowd\\fig\\runid" + str(self.run_id)
        mkdirs(gif_path)

        colors = ['r', 'g', 'b', 'm', 'c', 'k', 'w', 'k']

        total_traj = 0
        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, seq_start_end,
                 obs_frames, pred_frames, map_path, inv_h_t) = batch
                total_traj += fut_traj.size(1)

                rng = range(0,56)
                rng = range(56,80)
                rng = range(80, 115)
                fig, ax = plt.subplots()
                ax.imshow(imageio.imread(map_path[rng[0]]))

                rng = range(0,56)
                for idx in rng:
                    obs_real = obs_traj[:, idx, :2]
                    obs_real = np.concatenate([obs_real, np.ones((self.obs_len, 1))], axis=1)
                    obs_pixel = np.matmul(obs_real, inv_h_t[idx])
                    obs_pixel /= np.expand_dims(obs_pixel[:, 2], 1)
                    obs_pixel[:, [1, 0]] = obs_pixel[:, [0, 1]]

                    # gt_real = fut_traj[:, idx, :2]
                    # gt_real = np.concatenate([gt_real, np.ones((self.pred_len, 1))], axis=1)
                    # gt_pixel = np.matmul(gt_real, inv_h_t[idx])
                    # gt_pixel /= np.expand_dims(gt_pixel[:, 2], 1)
                    # gt_pixel[:, [1, 0]] = gt_pixel[:, [0, 1]]
                    # gt_data = np.concatenate([obs_pixel, gt_pixel], 0)

                    ax.scatter(obs_pixel[:,1], obs_pixel[:,0], s=1, c='r')
                    # ax.scatter(gt_pixel[:,1], gt_pixel[:,0], s=1, c='r')
                    # ax.scatter(gt_data[0,0], gt_data[0,1], s=5)


    def plot_traj_var(self, data_loader, num_samples=20):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        colors = ['r', 'g', 'b', 'm', 'c', 'k', 'w', 'k']

        total_traj = 0
        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, seq_start_end,
                 obs_frames, pred_frames, map_path, inv_h_t,
                 local_map, local_ic, local_homo) = batch
                batch_size = obs_traj.size(1)
                total_traj += fut_traj.size(1)



                #### MAP ####
                # for j, (s, e) in enumerate(seq_start_end):
                for (s, e) in seq_start_end:
                    agent_rng = range(s, e)
                    seq_map = imageio.imread(map_path[s])  # seq = 한 씬에서 모든 neighbors니까. 같은 데이터셋.

                    local_maps = []
                    goal20 = []
                    for idx in agent_rng:
                        map = imageio.imread(map_path[idx]) / 255
                        map = ndimage.distance_transform_edt(map)

                        obs_real = obs_traj[:, idx, :2].cpu().detach().numpy()
                        obs_real = np.concatenate([obs_real, np.ones((self.obs_len, 1))], axis=1)
                        obs_pixel = np.matmul(obs_real, inv_h_t[idx])
                        obs_pixel /= np.expand_dims(obs_pixel[:, 2], 1)
                        obs_pixel = obs_pixel[:, :2]
                        obs_pixel[:, [1, 0]] = obs_pixel[:, [0, 1]]

                        per_step_dist = (((obs_pixel[1:, :2] - obs_pixel[:-1, :2]) ** 2).sum(1) ** (1 / 2)).mean()
                        circle = np.zeros(map.shape)
                        for x in range(map.shape[0]):
                            for y in range(map.shape[1]):
                                dist_from_last_obs = np.linalg.norm([x, y] - obs_pixel[-1])
                                if dist_from_last_obs < per_step_dist * (12 + 1):
                                    angle = theta(([x, y] - (obs_pixel[-1] - obs_pixel[-2])) - obs_pixel[-2],
                                                  obs_pixel[-1] - obs_pixel[-2])
                                    if np.cos(angle) >= 0:
                                        circle[x, y] = np.cos(angle) * (
                                            1 + dist_from_last_obs) + 1  # in case dist_from_last_obs < 1

                        ##### find 20 goals
                        candidate_pos_ic = np.array(np.where(circle * map > 0)).transpose((1, 0))
                        if len(candidate_pos_ic) == 0:

                            avg_mvmt = np.abs((obs_pixel[1:, :2] - obs_pixel[:-1, :2]).mean(0))
                            rand_x = np.random.uniform(low=-avg_mvmt[0], high=avg_mvmt[0], size=(20,))
                            rand_y = np.random.uniform(low=-avg_mvmt[1], high=avg_mvmt[1], size=(20,))

                            selected_goal_ic = np.array([obs_pixel[-1]]*20) + np.vstack([rand_x, rand_y]).transpose((1,0))
                        else:
                            radius = per_step_dist * (self.pred_len + 1) / self.radius_deno
                            selected_goal_ic = find_coord(circle * map, circle * map, [], candidate_pos_ic, radius,
                                                          n_goal=20)
                            selected_goal_ic = np.array(selected_goal_ic)

                        fig, ax = plt.subplots()
                        ax.imshow(circle * map)
                        for coord in selected_goal_ic:
                            ax.scatter(coord[0], coord[1], s=1, c='hotpink', marker='x')
                        ax.scatter(obs_pixel[:, 1], obs_pixel[:, 0], s=1, c='b')


                        #### back to WCS goal
                        selected_goal_ic[:, [1, 0]] = selected_goal_ic[:, [0, 1]]
                        selected_goal_ic = np.concatenate([selected_goal_ic, np.ones((len(selected_goal_ic), 1))],
                                                          axis=1)
                        goal_wc = np.matmul(selected_goal_ic, np.linalg.inv(inv_h_t[idx]))
                        goal_wc = goal_wc / np.expand_dims(goal_wc[:, 2], 1)
                        goal20.append(goal_wc[:,:2])

                        plt.scatter(obs_traj[:, idx, 0], obs_traj[:, idx, 1], c='b')
                        plt.scatter(fut_traj[:, idx, 0], fut_traj[:, idx, 1], c='r')
                        plt.scatter(goal_wc[:, 0], goal_wc[:, 1], c='g', marker='X')

                        ##### resize the map
                        global_map = circle * map
                        local_map = transforms.Compose([
                            transforms.Resize(self.map_size),
                            transforms.ToTensor()
                        ])(Image.fromarray(global_map))
                        local_maps.append(local_map)



    def plot_gif(self, data_loader, num_samples=20):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        colors = ['r', 'g', 'b', 'm', 'c', 'k', 'w', 'k']

        total_traj = 0
        with torch.no_grad():
            b = 0
            for batch in data_loader:
                b += 1
                (obs_traj, fut_traj, seq_start_end,
                 obs_frames, pred_frames, map_path, inv_h_t,
                 local_map, local_ic, local_homo) = batch
                total_traj += fut_traj.size(1)

                #### MAP ####
                # for j, (s, e) in enumerate(seq_start_end):
                for (s, e) in seq_start_end:
                    agent_rng = range(s, e)
                    seq_map = imageio.imread(map_path[s])  # seq = 한 씬에서 모든 neighbors니까. 같은 데이터셋.


                    for dist in fut_rel_pos_dist20:
                        pred_fut_traj_rel = dist.rsample()
                        pred_fut_traj = integrate_samples(pred_fut_traj_rel, obs_traj[-1, :, :2],
                                                          dt=self.dt)

                        gt_data, pred_data = [], []

                        for j in range(len(agent_rng)):
                            one_ped = agent_rng[j]
                            obs_real = obs_traj[:, one_ped, :2]
                            obs_real = np.concatenate([obs_real, np.ones((self.obs_len, 1))], axis=1)
                            obs_pixel = np.matmul(obs_real, inv_h_t[j])
                            obs_pixel /= np.expand_dims(obs_pixel[:, 2], 1)
                            obs_pixel[:, [1, 0]] = obs_pixel[:, [0, 1]]

                            gt_real = fut_traj[:, one_ped, :2]
                            gt_real = np.concatenate([gt_real, np.ones((self.pred_len, 1))], axis=1)
                            gt_pixel = np.matmul(gt_real, inv_h_t[j])
                            gt_pixel /= np.expand_dims(gt_pixel[:, 2], 1)
                            gt_pixel[:, [1, 0]] = gt_pixel[:, [0, 1]]
                            gt_data.append(np.concatenate([obs_pixel, gt_pixel], 0))  # (20, 3)

                            pred_real = pred_fut_traj[:, one_ped].numpy()
                            pred_pixel = np.concatenate([pred_real, np.ones((self.pred_len, 1))],
                                                        axis=1)
                            pred_pixel = np.matmul(pred_pixel, inv_h_t[j])
                            pred_pixel /= np.expand_dims(pred_pixel[:, 2], 1)
                            pred_pixel[:, [1, 0]] = pred_pixel[:, [0, 1]]

                            pred_data.append(np.concatenate([obs_pixel, pred_pixel], 0))

                        gt_data = np.stack(gt_data)
                        pred_data = np.stack(pred_data)

                        multi_sample_pred.append(pred_data)

                    def init():
                        ax.imshow(seq_map)

                    def update_dot(num_t):
                        print(num_t)
                        ax.imshow(seq_map)

                        for i in range(n_agent):
                            ln_gt[i].set_data(gt_data[i, :num_t, 1], gt_data[i, :num_t, 0])
                            for j in range(20):
                                all_ln_pred[i][j].set_data(multi_sample_pred[j][i, :num_t, 1],
                                                           multi_sample_pred[j][i, :num_t, 0])

                    n_agent = gt_data.shape[0]
                    n_frame = gt_data.shape[1]

                    fig, ax = plt.subplots()
                    title = map_path[j].split('.')[0].split('\\')[-1].replace('/', '_')
                    ax.set_title(title, fontsize=9)
                    fig.tight_layout()

                    ln_gt = []
                    all_ln_pred = []

                    for i in range(n_agent):
                        ln_gt.append(ax.plot([], [], colors[i % len(colors)] + '--', linewidth=1)[0])
                        # ln_gt.append(ax.scatter([], [], c=colors[i % len(colors)], s=2))

                        ln_pred = []
                        for _ in range(20):
                            ln_pred.append(
                                ax.plot([], [], colors[i % len(colors)], alpha=0.6, linewidth=1)[0])
                            ln_pred.append(
                                ax.plot([], [], colors[i % len(colors)], alpha=0.6, linewidth=1)[0])
                        all_ln_pred.append(ln_pred)

                    ani = FuncAnimation(fig, update_dot, frames=n_frame, interval=1, init_func=init())

                    # writer = PillowWriter(fps=3000)
                    gif_path = 'D:\crowd\datasets\Trajectories'
                    ani.save(gif_path + "/" + self.dataset_name + "_" + title + "_agent" + str(
                        agent_rng[0]) + "to" + str(agent_rng[-1]) + ".gif", fps=4)

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
        test_map_coll = torch.Tensor(data['test_map_coll'])
        map_coll = torch.Tensor(data['map_coll'])

        test_loss_recon = torch.Tensor(data['test_loss_recon'])
        test_loss_kl = torch.Tensor(data['test_loss_kl'])

        test_loss_coll = torch.Tensor(data['test_loss_coll'])
        loss_coll = torch.Tensor(data['loss_coll'])
        total_coll = torch.Tensor(data['total_coll'])
        test_total_coll = torch.Tensor(data['test_total_coll'])

        self.viz.line(
            X=iters, Y=map_coll, env=self.name + '/lines',
            win=self.win_id['map_coll'], update='append',
            opts=dict(xlabel='iter', ylabel='map_coll',
                      title='map_coll')
        )

        self.viz.line(
            X=iters, Y=test_map_coll, env=self.name + '/lines',
            win=self.win_id['test_map_coll'], update='append',
            opts=dict(xlabel='iter', ylabel='test_map_coll',
                      title='test_map_coll')
        )


        self.viz.line(
            X=iters, Y=total_coll, env=self.name + '/lines',
            win=self.win_id['total_coll'], update='append',
            opts=dict(xlabel='iter', ylabel='total_coll',
                      title='total_coll')
        )

        self.viz.line(
            X=iters, Y=test_total_coll, env=self.name + '/lines',
            win=self.win_id['test_total_coll'], update='append',
            opts=dict(xlabel='iter', ylabel='test_total_coll',
                      title='test_total_coll')
        )


        self.viz.line(
            X=iters, Y=test_loss_coll, env=self.name + '/lines',
            win=self.win_id['test_loss_coll'], update='append',
            opts=dict(xlabel='iter', ylabel='test_loss_coll',
                      title='test_loss_coll')
        )

        self.viz.line(
            X=iters, Y=loss_coll, env=self.name + '/lines',
            win=self.win_id['loss_coll'], update='append',
            opts=dict(xlabel='iter', ylabel='loss_coll',
                      title='loss_coll')
        )



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