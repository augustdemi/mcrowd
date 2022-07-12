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
        self.name = '%s_bs%s_dr_mlp_%s_dr_rnn_%s_enc_hD_%s_dec_hD_%s_lr_%s_scale_%s_num_sg_%s' \
                    'ctxtD_%s_coll_th_%s_w_coll_%s_beta_%s_lr_e_%s' % \
                    (args.dataset_name, args.batch_size, args.dropout_mlp, args.dropout_rnn, args.encoder_h_dim,
                     args.decoder_h_dim, args.lr_VAE, args.scale, args.num_sg, args.context_dim, args.coll_th, args.w_coll, args.beta, args.lr_e)

        # to be appended by run_id

        # self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = args.device
        self.dt=0.5
        self.eps=1e-9
        self.ll_prior_w =args.ll_prior_w
        self.sg_idx = np.array(range(12))
        self.sg_idx = np.flip(11-self.sg_idx[::(12//args.num_sg)])

        self.coll_th = args.coll_th
        self.beta = args.beta
        self.context_dim = args.context_dim
        self.w_coll = args.w_coll


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



        # create dirs: "records", "ckpts", "outputs" (if not exist)
        mkdirs("records");
        mkdirs("ckpts");
        mkdirs("outputs")

        # set run id
        self.run_id = args.run_id


        # finalize name
        self.name = self.name + '_run_' + str(self.run_id)

        # checkpoints
        self.ckpt_dir = os.path.join("ckpts", self.name)

        # visdom setup
        self.viz_on = args.viz_on
        if self.viz_on:
            self.win_id = dict(
                recon='win_recon', loss_kl='win_loss_kl', loss_recon='win_loss_recon',
                ade_min='win_ade_min', fde_min='win_fde_min', ade_avg='win_ade_avg', fde_avg='win_fde_avg',
                ade_std='win_ade_std', fde_std='win_fde_std',
                test_loss_recon='win_test_loss_recon', test_loss_kl='win_test_loss_kl',
                loss_recon_prior='win_loss_recon_prior', loss_coll='win_loss_coll', test_loss_coll = 'win_test_loss_coll',
                test_total_coll='win_test_total_coll', total_coll='win_total_coll'
            )
            self.line_gather = DataGather(
                'iter', 'loss_recon', 'loss_kl',  'loss_recon_prior',
                'ade_min', 'fde_min', 'ade_avg', 'fde_avg', 'ade_std', 'fde_std',
                'test_loss_recon', 'test_loss_kl', 'test_loss_coll', 'loss_coll', 'test_total_coll', 'total_coll'
            )


            self.viz_port = args.viz_port  # port number, eg, 8097
            self.viz = visdom.Visdom(port=self.viz_port, env=self.name)
            self.viz_ll_iter = args.viz_ll_iter
            self.viz_la_iter = args.viz_la_iter

            self.viz_init()
        #### create a new model or load a previously saved model

        self.ckpt_load_iter = args.ckpt_load_iter

        self.obs_len = 4
        self.pred_len = 12
        self.num_layers = args.num_layers
        self.decoder_h_dim = args.decoder_h_dim

        if self.ckpt_load_iter == 0 or args.dataset_name =='all':  # create a new model
            lg_cvae_path = 'nu.lgcvae_enc_block_1_fcomb_block_2_wD_10_lr_0.0001_lg_klw_1.0_a_0.25_r_2.0_fb_3.0_anneal_e_10_aug_1_llprior_0.0_run_4'
            lg_cvae_path = os.path.join('ckpts', lg_cvae_path, 'iter_39000_lg_cvae.pt')

            if self.device == 'cuda':
                self.lg_cvae = torch.load(lg_cvae_path)

            self.decoderMy = Decoder(
                args.pred_len,
                dec_h_dim=self.decoder_h_dim,
                enc_h_dim=args.encoder_h_dim,
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
        params = list(self.decoderMy.parameters())
        # create optimizers
        self.optimizer = optim.Adam(
            params,
            lr=self.lr
        )

        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                        lr_lambda=lambda epoch: args.lr_e ** epoch)

        print('Start loading data...')

        if self.ckpt_load_iter != self.max_iter:
            cfg = Config('nuscenes_train', False, create_dirs=True)
            torch.set_default_dtype(torch.float32)
            log = open('log' + self.name + '.txt', 'a+')
            self.train_loader = data_generator(cfg, log, split='train', phase='training',
                                               batch_size=args.batch_size, device=self.device, scale=args.scale, shuffle=True)

            cfg = Config('nuscenes', False, create_dirs=True)
            torch.set_default_dtype(torch.float32)
            log = open('log' + self.name + '.txt', 'a+')
            self.val_loader = data_generator(cfg, log, split='test', phase='testing',
                                             batch_size=args.batch_size, device=self.device, scale=args.scale, shuffle=True)

            print(
                'There are {} iterations per epoch'.format(len(self.train_loader.idx_list))
            )
        print('...done')

    def temmp(self):
        aa = torch.zeros((120, 2, 256, 256)).to(self.device)
        self.lg_cvae.unet.down_forward(aa)


    ####
    def train(self):
        self.set_mode(train=True)
        data_loader = self.train_loader

        iter_per_epoch = len(data_loader.idx_list)
        start_iter = self.ckpt_load_iter + 1
        epoch = int(start_iter / iter_per_epoch) + 1

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
                if epoch % 10 == 0:
                    if self.optimizer.param_groups[0]['lr'] > 1e-4:
                        self.scheduler.step()
                    else:
                        self.optimizer.param_groups[0]['lr'] = 1e-4
                print("lr: ", self.optimizer.param_groups[0]['lr'], ' // w_coll: ', self.w_coll)
                print('e_coll_loss: ', e_coll_loss, ' // e_total_coll: ', e_total_coll)

                epoch +=1
                prev_e_coll_loss = e_coll_loss
                prev_e_total_coll = e_total_coll
                e_coll_loss = 0
                e_total_coll = 0

            # ============================================
            #          TRAIN THE VAE (ENC & DEC)
            # ============================================

            (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
             maps, local_map, local_ic, local_homo) = data
            batch_size = fut_traj.size(1) #=sum(seq_start_end[:,1] - seq_start_end[:,0])

            # TF, goals, z~posterior
            fut_rel_pos_dist_tf = self.decoderMy(
                seq_start_end,
                obs_traj_st[-1],
                obs_traj[-1, :, :2],
                fut_traj[list(self.sg_idx), :, :2].permute(1,0,2), # goal
                self.sg_idx,
                fut_vel_st, # TF
                train=True
            )


            ll_tf = torch.abs(fut_rel_pos_dist_tf - fut_vel_st).sum().div(batch_size)
            traj_elbo = ll_tf



            coll_loss = torch.tensor(0.0).to(self.device)
            total_coll = 0
            n_scene = 0

            if self.w_coll > 0:
                pred_fut_traj = integrate_samples(fut_rel_pos_dist_tf * self.scale,
                                                       obs_traj[-1, :, :2],
                                                       dt=self.dt)
                for s, e in seq_start_end:
                    n_scene += 1
                    num_ped = e - s
                    if num_ped == 1:
                        continue
                    for t in range(self.pred_len):
                        ## posterior
                        curr1_post = pred_fut_traj[t, s:e].repeat(num_ped, 1)
                        curr2_post = self.repeat(pred_fut_traj[t, s:e], num_ped)
                        dist_post = torch.norm(curr1_post - curr2_post, dim=1)
                        dist_post = dist_post.reshape(num_ped, num_ped)
                        diff_agent_dist_post = dist_post[torch.where(dist_post > 0)]
                        coll_loss += (torch.sigmoid(-(diff_agent_dist_post - self.coll_th) * self.beta)).sum()
                        total_coll += (len(torch.where(diff_agent_dist_post < 1.5)[0]) / 2)

            coll_loss = coll_loss.div(batch_size)
            total_coll = total_coll/batch_size

            loss = traj_elbo + self.w_coll * coll_loss
            e_coll_loss +=coll_loss.item()
            e_total_coll +=total_coll

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            # save model parameters
            if epoch > 50 and (iteration % (iter_per_epoch*50) == 0) and (epoch < 500):
                self.save_checkpoint(epoch)

            # (visdom) insert current line stats
            if epoch > 0:
                if (self.viz_on and (iteration % (iter_per_epoch*500) == 0)):
                    ade_min, fde_min, \
                    ade_avg, fde_avg, \
                    ade_std, fde_std, \
                    test_loss_recon, test_loss_kl, test_loss_coll, test_total_coll = self.evaluate_dist(self.val_loader, loss=True)
                    self.line_gather.insert(iter=epoch,
                                            ade_min=ade_min,
                                            fde_min=fde_min,
                                            ade_avg=ade_avg,
                                            fde_avg=fde_avg,
                                            ade_std=ade_std,
                                            fde_std=fde_std,
                                            loss_recon=ll_tf.item(),
                                            loss_recon_prior=0,
                                            loss_kl=0,
                                            loss_coll=prev_e_coll_loss,
                                            total_coll=prev_e_total_coll,
                                            test_loss_recon=test_loss_recon.item(),
                                            test_loss_kl=0,
                                            test_loss_coll=test_loss_coll.item(),
                                            test_total_coll=test_total_coll
                                            )
                    prn_str = ('[iter_%d (epoch_%d)] vae_loss: %.3f ' + \
                                  '(recon: %.3f, kl: %.3f)\n' + \
                                  'ADE min: %.2f, FDE min: %.2f, ADE avg: %.2f, FDE avg: %.2f\n'
                              ) % \
                              (iteration, epoch,
                               loss.item(), 0, 0,
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

                fut_rel_pos_dist = self.decoderMy(
                    seq_start_end,
                    obs_traj_st[-1],
                    obs_traj[-1, :, :2],
                    fut_traj[list(self.sg_idx), :, :2].permute(1, 0, 2),  # goal
                    self.sg_idx,
                )

                pred_fut_traj = integrate_samples(fut_rel_pos_dist * self.scale,
                                                  obs_traj[-1, :, :2],
                                                  dt=self.dt)
                for s, e in seq_start_end:
                    n_scene +=1
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
                        if len(diff_agent_dist) > 0:
                            # diff_agent_dist[torch.where(diff_agent_dist > self.coll_th)] += self.beta
                            coll_loss += (torch.sigmoid(-(diff_agent_dist - self.coll_th) * self.beta)).sum().div(batch_size)
                            total_coll += (len(torch.where(diff_agent_dist < 1.5)[0])/2) / batch_size

                ade, fde = [], []
                ade.append(displacement_error(
                    pred_fut_traj, fut_traj[:,:,:2], mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_fut_traj[-1], fut_traj[-1,:,:2], mode='raw'
                ))
                all_ade.append(torch.stack(ade))
                all_fde.append(torch.stack(fde))
                loss_recon += torch.abs(fut_rel_pos_dist-fut_vel_st).sum().div(batch_size)

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
                   loss_recon/b, loss_kl/b, coll_loss/b, total_coll
        else:
            return ade_min, fde_min, \
                   ade_avg, fde_avg, \
                   ade_std, fde_std,

    def collision_stat(self, data_loader):
        self.set_mode(train=False)

        total_coll1 = 0
        total_coll2 = 0
        total_coll3 = 0
        total_coll4 = 0
        total_coll5 = 0
        total_coll6 = 0
        n_scene = 0
        total_ped = []
        e_ped = []
        avg_dist = 0
        min_dist = 10000
        n_agent = 0

        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
                 obs_frames, fut_frames, map_path, inv_h_t,
                 local_map, local_ic, local_homo) = batch
                for s, e in seq_start_end:
                    n_scene +=1
                    num_ped = e - s
                    total_ped.append(num_ped)
                    if num_ped == 1:
                        continue
                    e_ped.append(num_ped)

                    seq_traj = fut_traj[:,s:e,:2]
                    for i in range(len(seq_traj)):
                        curr1 = seq_traj[i].repeat(num_ped, 1)
                        curr2 = self.repeat(seq_traj[i], num_ped)
                        dist = torch.sqrt(torch.pow(curr1 - curr2, 2).sum(1)).cpu().numpy()
                        dist = dist.reshape(num_ped, num_ped)
                        diff_agent_idx = np.triu_indices(num_ped, k=1)
                        diff_agent_dist = dist[diff_agent_idx]
                        avg_dist += diff_agent_dist.sum()
                        min_dist = min(min_dist, diff_agent_dist.min())
                        n_agent += len(diff_agent_dist)
                        total_coll1 += (diff_agent_dist < 0.05).sum()
                        total_coll2 += (diff_agent_dist < 0.1).sum()
                        total_coll3 += (diff_agent_dist < 0.2).sum()
                        total_coll4 += (diff_agent_dist < 0.3).sum()
                        total_coll5 += (diff_agent_dist < 0.4).sum()
                        total_coll6 += (diff_agent_dist < 0.5).sum()
        print('total_coll1: ', total_coll1)
        print('total_coll2: ', total_coll2)
        print('total_coll3: ', total_coll3)
        print('total_coll4: ', total_coll4)
        print('total_coll5: ', total_coll5)
        print('total_coll6: ', total_coll6)
        print('n_scene: ', n_scene)
        print('e_ped:', len(e_ped))
        print('total_ped:', len(total_ped))
        print('avg_dist:', avg_dist/n_agent)
        print('min_dist:', avg_dist/n_agent)
        print('e_ped:', np.array(e_ped).mean())
        print('total_ped:', np.array(total_ped).mean())



    ####
    def viz_init(self):
        self.viz.close(env=self.name , win=self.win_id['loss_recon'])
        self.viz.close(env=self.name , win=self.win_id['loss_recon_prior'])
        self.viz.close(env=self.name , win=self.win_id['loss_kl'])
        self.viz.close(env=self.name , win=self.win_id['test_loss_recon'])
        self.viz.close(env=self.name , win=self.win_id['test_loss_kl'])

        self.viz.close(env=self.name , win=self.win_id['ade_min'])
        self.viz.close(env=self.name , win=self.win_id['fde_min'])
        self.viz.close(env=self.name , win=self.win_id['ade_avg'])
        self.viz.close(env=self.name , win=self.win_id['fde_avg'])
        self.viz.close(env=self.name , win=self.win_id['ade_std'])
        self.viz.close(env=self.name , win=self.win_id['fde_std'])
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
        test_loss_coll = torch.Tensor(data['test_loss_coll'])
        loss_coll = torch.Tensor(data['loss_coll'])
        total_coll = torch.Tensor(data['total_coll'])
        test_total_coll = torch.Tensor(data['test_total_coll'])

        self.viz.line(
            X=iters, Y=total_coll, env=self.name ,
            win=self.win_id['total_coll'], update='append',
            opts=dict(xlabel='iter', ylabel='total_coll',
                      title='total_coll')
        )

        self.viz.line(
            X=iters, Y=test_total_coll, env=self.name ,
            win=self.win_id['test_total_coll'], update='append',
            opts=dict(xlabel='iter', ylabel='test_total_coll',
                      title='test_total_coll')
        )


        self.viz.line(
            X=iters, Y=test_loss_coll, env=self.name ,
            win=self.win_id['test_loss_coll'], update='append',
            opts=dict(xlabel='iter', ylabel='test_loss_coll',
                      title='test_loss_coll')
        )

        self.viz.line(
            X=iters, Y=loss_coll, env=self.name ,
            win=self.win_id['loss_coll'], update='append',
            opts=dict(xlabel='iter', ylabel='loss_coll',
                      title='loss_coll')
        )
        self.viz.line(
            X=iters, Y=loss_recon, env=self.name ,
            win=self.win_id['loss_recon'], update='append',
            opts=dict(xlabel='iter', ylabel='-loglikelihood',
                      title='Recon. loss of predicted future traj')
        )

        self.viz.line(
            X=iters, Y=loss_recon_prior, env=self.name ,
            win=self.win_id['loss_recon_prior'], update='append',
            opts=dict(xlabel='iter', ylabel='-loglikelihood',
                      title='Recon. loss - prior')
        )


        self.viz.line(
            X=iters, Y=loss_kl, env=self.name ,
            win=self.win_id['loss_kl'], update='append',
            opts=dict(xlabel='iter', ylabel='kl divergence',
                      title='KL div. btw posterior and c. prior'),
        )


        self.viz.line(
            X=iters, Y=test_loss_recon, env=self.name ,
            win=self.win_id['test_loss_recon'], update='append',
            opts=dict(xlabel='iter', ylabel='-loglikelihood',
                      title='Test Recon. loss of predicted future traj')
        )

        self.viz.line(
            X=iters, Y=test_loss_kl, env=self.name ,
            win=self.win_id['test_loss_kl'], update='append',
            opts=dict(xlabel='iter', ylabel='kl divergence',
                      title='Test KL div. btw posterior and c. prior'),
        )


        self.viz.line(
            X=iters, Y=ade_min, env=self.name ,
            win=self.win_id['ade_min'], update='append',
            opts=dict(xlabel='iter', ylabel='ade',
                      title='ADE min'),
        )
        self.viz.line(
            X=iters, Y=fde_min, env=self.name ,
            win=self.win_id['fde_min'], update='append',
            opts=dict(xlabel='iter', ylabel='fde',
                      title='FDE min'),
        )
        self.viz.line(
            X=iters, Y=ade_avg, env=self.name ,
            win=self.win_id['ade_avg'], update='append',
            opts=dict(xlabel='iter', ylabel='ade',
                      title='ADE avg'),
        )

        self.viz.line(
            X=iters, Y=fde_avg, env=self.name ,
            win=self.win_id['fde_avg'], update='append',
            opts=dict(xlabel='iter', ylabel='fde',
                      title='FDE avg'),
        )
        self.viz.line(
            X=iters, Y=ade_std, env=self.name ,
            win=self.win_id['ade_std'], update='append',
            opts=dict(xlabel='iter', ylabel='ade std',
                      title='ADE std'),
        )

        self.viz.line(
            X=iters, Y=fde_std, env=self.name ,
            win=self.win_id['fde_std'], update='append',
            opts=dict(xlabel='iter', ylabel='fde std',
                      title='FDE std'),
        )


    def set_mode(self, train=True):

        if train:
            self.decoderMy.train()
        else:
            self.decoderMy.eval()

    ####
    def save_checkpoint(self, iteration):

        decoderMy_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_decoderMy.pt' % iteration
        )
        mkdirs(self.ckpt_dir)

        torch.save(self.decoderMy, decoderMy_path)
    ####
    def load_checkpoint(self):

        decoderMy_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_decoderMy.pt' % self.ckpt_load_iter
        )

        if self.device == 'cuda':
            self.decoderMy = torch.load(decoderMy_path)
        else:
            self.decoderMy = torch.load(decoderMy_path, map_location='cpu')