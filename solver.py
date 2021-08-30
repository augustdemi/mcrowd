import os

import torch.optim as optim
# -----------------------------------------------------------------------------#
from utils import DataGather, mkdirs
from model import *
from loss import kl_two_gaussian, displacement_error, final_displacement_error
from data.loader import data_loader
import imageio
import visdom

import matplotlib.pyplot as plt
from torch.distributions import RelaxedOneHotCategorical as concrete
from torch.distributions import OneHotCategorical as discrete
from torch.distributions import kl_divergence
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
# from model_map_ae import Decoder as Map_Decoder

import numpy as np

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


class Solver(object):

    ####
    def __init__(self, args):

        self.args = args

        # self.name = '%s_pred_len_%s_zS_%s_embedding_dim_%s_enc_h_dim_%s_dec_h_dim_%s_mlp_dim_%s_pool_dim_%s_lr_%s_klw_%s' % \
        #             (args.dataset_name, args.pred_len, args.zS_dim, 16, args.encoder_h_dim, args.decoder_h_dim, args.mlp_dim, args.pool_dim, args.lr_VAE, args.kl_weight)

        self.name = '%s_pred_len_%s_zS_%s_dr_mlp_%s_dr_rnn_%s_enc_h_dim_%s_dec_h_dim_%s_mlp_dim_%s_map_feat_dim_%s_lr_%s_klw_%s_klgw_%s_gvaew_%s_g_mlp_%s' % \
                    (args.dataset_name, args.pred_len, args.zS_dim, args.dropout_mlp, args.dropout_rnn, args.encoder_h_dim,
                     args.decoder_h_dim, args.mlp_dim, args.map_feat_dim , args.lr_VAE, args.kl_weight, args.kl_weight_goal, args.goal_vae_w, args.goal_mlp_dim)

        # self.name = '%s_pred_len_%s_zS_%s_dr_mlp_%s_dr_rnn_%s_enc_h_dim_%s_dec_h_dim_%s_mlp_dim_%s_attn_%s_lr_%s_klw_%s' % \
        #             (args.dataset_name, args.pred_len, args.zS_dim, args.dropout_mlp, args.dropout_rnn, args.encoder_h_dim,
        #              args.decoder_h_dim, args.mlp_dim, args.attention, args.lr_VAE, args.kl_weight)


        # to be appended by run_id

        # self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = args.device
        self.temp=1.99
        self.goal_temp=0.66
        self.dt=0.4
        self.eps=1e-9
        self.n_grid=256

        self.kl_weight=args.kl_weight
        self.kl_weight_goal=args.kl_weight_goal
        self.goal_vae_w=args.goal_vae_w

        self.max_iter = int(args.max_iter)
        self.map_size = args.map_size


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
        self.lr_VAE = args.lr_VAE
        self.beta1_VAE = args.beta1_VAE
        self.beta2_VAE = args.beta2_VAE
        print(args.desc)

        # visdom setup
        self.viz_on = args.viz_on
        if self.viz_on:
            self.win_id = dict(
                recon='win_recon', loss_kl='win_loss_kl', loss_recon='win_loss_recon', total_loss='win_total_loss',
                loss_map='win_loss_map', loss_vel='win_loss_vel', test_loss_map='win_test_loss_map', test_loss_vel='win_test_loss_vel',
                ade_min='win_ade_min', fde_min='win_fde_min', ade_avg='win_ade_avg', fde_avg='win_fde_avg',
                ade_std='win_ade_std', fde_std='win_fde_std',
                test_loss_recon='win_test_loss_recon', test_loss_kl='win_test_loss_kl', test_total_loss='win_test_total_loss',
                loss_recon_prior='win_loss_recon_prior', loss_recon_goal='win_loss_recon_goal',
                loss_recon_goal_prior='win_loss_recon_goal_prior',
                loss_kl_goal='win_loss_kl_goal', loss_kl_goal_prior='win_loss_kl_goal_prior'
            )


            self.line_gather = DataGather(
                'iter', 'loss_recon', 'loss_kl', 'total_loss', 'ade_min', 'fde_min',
                'ade_avg', 'fde_avg', 'ade_std', 'fde_std',
                'test_loss_recon', 'test_loss_kl', 'test_total_loss',
                'test_loss_map', 'test_loss_vel', 'loss_map', 'loss_vel',
                'loss_recon_prior', 'loss_recon_goal', 'loss_recon_goal_prior', 'loss_kl_goal', 'loss_kl_goal_prior'
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

        self.obs_len = args.obs_len
        self.pred_len = args.pred_len
        self.num_layers = args.num_layers
        self.decoder_h_dim = args.decoder_h_dim

        if self.ckpt_load_iter == 0 or args.dataset_name =='all':  # create a new model
            self.load_map_encoder()
            self.encoderMx = Encoder(
                self.map_encoder,
                args.zS_dim,
                enc_h_dim=args.encoder_h_dim,
                mlp_dim=args.mlp_dim,
                batch_norm=args.batch_norm,
                num_layers=args.num_layers,
                dropout_mlp=args.dropout_mlp,
                dropout_rnn=args.dropout_rnn,
                map_feat_dim=args.map_feat_dim,
                device=self.device).to(self.device)
            self.encoderMy = EncoderY(
                self.map_encoder,
                args.zS_dim,
                enc_h_dim=args.encoder_h_dim,
                mlp_dim=args.mlp_dim,
                batch_norm=args.batch_norm,
                num_layers=args.num_layers,
                dropout_mlp=args.dropout_mlp,
                dropout_rnn=args.dropout_rnn,
                map_feat_dim=args.map_feat_dim,
                device=self.device).to(self.device)
            self.decoderMy = Decoder(
                self.map_encoder,
                args.pred_len,
                dec_h_dim=self.decoder_h_dim,
                enc_h_dim=args.encoder_h_dim,
                mlp_dim=args.mlp_dim,
                z_dim=args.zS_dim,
                num_layers=args.num_layers,
                device=args.device,
                dropout_rnn=args.dropout_rnn,
                map_feat_dim=args.map_feat_dim,
                map_size=args.map_size).to(self.device)

            self.encoderMx_goal = EncoderX_Goal(
                args.w_dim,
                enc_h_dim=args.goal_mlp_dim).to(self.device)
            self.encoderM_goal = Encoder_Goal(
                args.w_dim,
                enc_h_dim=args.goal_mlp_dim).to(self.device)

            self.decoderM_goal = Decoder_Goal(
                args.w_dim,
                enc_h_dim=args.goal_mlp_dim).to(self.device)

        else:  # load a previously saved model
            print('Loading saved models (iter: %d)...' % self.ckpt_load_iter)
            self.load_checkpoint()
            print('...done')


        # get VAE parameters
        vae_params = \
            list(self.encoderMx.parameters()) + \
            list(self.encoderMy.parameters()) + \
            list(self.decoderMy.parameters()) + \
            list(self.encoderMx_goal.parameters()) + \
            list(self.encoderM_goal.parameters()) + \
            list(self.decoderM_goal.parameters()) + \
            list(self.map_encoder.parameters()) + \
            list(self.map_decoder.parameters())

        # create optimizers
        self.optim_vae = optim.Adam(
            vae_params,
            lr=self.lr_VAE,
            betas=[self.beta1_VAE, self.beta2_VAE]
        )

        # prepare dataloader (iterable)
        print('Start loading data...')
        # args.batch_size=4
        # self.agrs = args
        train_path = os.path.join(self.dataset_dir, self.dataset_name, 'Train.txt')
        val_path = os.path.join(self.dataset_dir, self.dataset_name, 'Test.txt')

        # long_dtype, float_dtype = get_dtypes(args)

        if self.ckpt_load_iter != self.max_iter:
            print("Initializing train dataset")
            _, self.train_loader = data_loader(self.args, train_path)
            print("Initializing val dataset")
            _, self.val_loader = data_loader(self.args, val_path)

            print(
                'There are {} iterations per epoch'.format(len(self.train_loader.dataset) / args.batch_size)
            )
        print('...done')



    ####
    def train(self):
        self.set_mode(train=True)

        data_loader = self.train_loader
        self.N = len(data_loader.dataset)

        # iterators from dataloader
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

            # sample a mini-batch
            (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
             obs_frames, pred_frames, obs_obst, fut_obst, map_path, inv_h_t, obs_heatmap, goal_heatmap) = next(iterator)
            batch = obs_traj.size(1) #=sum(seq_start_end[:,1] - seq_start_end[:,0])


            ### Trajectory Encoders ###
            (encX_h_feat, logitX, map_featX) \
                = self.encoderMx(obs_traj_st, seq_start_end, obs_obst, obs_traj[:,:,2:4], train=True)
            (_, logitY, map_featY) \
                = self.encoderMy(obs_traj_st[-1], fut_vel_st, seq_start_end, encX_h_feat, fut_obst, fut_traj[:,:,2:4],train=True)


            ### Goal VAE ###
            goal_encX_h_feat, logitX_goal = self.encoderMx_goal(obs_heatmap, train=True)
            logitY_goal = self.encoderM_goal(goal_encX_h_feat, goal_heatmap, train=True)
            # conditional prior
            p_dist_goal = discrete(logits=logitX_goal)
            # posterior
            q_dist_goal = discrete(logits=logitY_goal)
            relaxed_q_dist_goal = concrete(logits=logitY_goal, temperature=self.goal_temp)
            ## the discrete uniform distribution prior of the goal: it should always has the same probabilities for each 20 values: req_grad = False
            logit_unif_prior_goal = torch.zeros((batch, 20)).to(self.device)
            unif_prior_goal = discrete(logits=logit_unif_prior_goal)
            # relaxed_unif_prior_goal = concrete(logits=logit_unif_prior_goal, temperature=self.goal_temp)

            w_posterior = relaxed_q_dist_goal.rsample()
            goal_posterior = self.decoderM_goal(goal_encX_h_feat, w_posterior)

            ### Trajectory latent dist. and Decoder ###
            p_dist = discrete(logits=logitX)
            q_dist = discrete(logits=logitY)
            relaxed_q_dist = concrete(logits=logitY, temperature=self.temp)
            z = relaxed_q_dist.rsample()

            ##### Goal heatmap to traj #####
            recon_goal_heatmap = F.interpolate(goal_posterior, size=(224,224))
            # recon_goal_heatmap2 = F.interpolate(goal_heatmap, size=(224,224))
            pred_goal_pos_posterior = []
            for idx, (start, end) in enumerate(seq_start_end):
                for t in range(start, end):
                    # goal_idx = np.where(recon_goal_heatmap[t,0]==recon_goal_heatmap[t,0].max())
                    # pred_goal = np.array([goal_idx[1][len(goal_idx)//2], goal_idx[0][len(goal_idx)//2]])
                    # pred_goal = np.concatenate([pred_goal, [1]], axis=0)
                    # pred_goal = np.matmul(pred_goal, np.linalg.inv(inv_h_t[idx]))
                    x_goal_pixel = 0
                    y_goal_pixel = 0
                    exp_recon = torch.exp(recon_goal_heatmap[t, 0]*(30/recon_goal_heatmap[t].max()))
                    for pixel_idx in range(224):
                        x_goal_pixel += pixel_idx * exp_recon[pixel_idx, :].sum() / exp_recon.sum()
                        y_goal_pixel += pixel_idx * exp_recon[:, pixel_idx].sum() / exp_recon.sum()
                    goal_pixel = torch.cat([torch.round(y_goal_pixel).unsqueeze(0), torch.round(x_goal_pixel).unsqueeze(0), torch.ones(1)])
                    pred_goal = torch.matmul(goal_pixel, torch.tensor(np.linalg.inv(inv_h_t[idx])).float())
                    pred_goal_pos_posterior.append(pred_goal / pred_goal[2])
            pred_goal_pos_posterior = torch.stack(pred_goal_pos_posterior)[:,:2]

            fut_rel_pos_dist_posterior = self.decoderMy(
                obs_traj_st[-1],
                encX_h_feat,
                z, fut_traj[-1,:,:2],
                torch.cat((obs_obst[-1].unsqueeze(0), fut_obst), dim=0),
                fut_traj
            )

            ### Goal & Traj generation from the "prior" latents of goals ###
            ll_goal_prior = []
            loss_goal_pos_onehot = 0
            for w_one_hot in torch.eye(20)[torch.randperm(20)[:5]]:
                print(w_one_hot)

                w_one_hot = w_one_hot.unsqueeze(0).repeat((goal_encX_h_feat.shape[0], 1)).to(self.device)
                # predict goal from the goal prior
                goal_prior = self.decoderM_goal(goal_encX_h_feat, w_one_hot)
                ll_goal_prior.append(-F.l1_loss(goal_prior, goal_heatmap, reduction='none'))

                recon_goal_heatmap = F.interpolate(goal_prior, size=(224, 224))
                pred_goal_pos_prior = []
                for idx, (start, end) in enumerate(seq_start_end):
                    for t in range(start, end):
                        x_goal_pixel = 0
                        y_goal_pixel = 0
                        exp_recon = torch.exp(recon_goal_heatmap[t, 0]*(30/recon_goal_heatmap[t].max()))
                        for pixel_idx in range(224):
                            x_goal_pixel += pixel_idx * exp_recon[pixel_idx, :].sum() / exp_recon.sum()
                            y_goal_pixel += pixel_idx * exp_recon[:, pixel_idx].sum() / exp_recon.sum()
                        goal_pixel = torch.cat([torch.round(y_goal_pixel).unsqueeze(0), torch.round(x_goal_pixel).unsqueeze(0), torch.ones(1)])
                        pred_goal = torch.matmul(goal_pixel, torch.tensor(np.linalg.inv(inv_h_t[idx])).float())
                        pred_goal_pos_prior.append(pred_goal / pred_goal[2])
                pred_goal_pos_prior = torch.stack(pred_goal_pos_prior)[:,:2]
                loss_goal_pos_onehot += ((pred_goal_pos_prior - fut_traj[-1, :, :2]) ** 2).mean()



            ### Loss ###
            ll_goal_prior = torch.stack(ll_goal_prior).sum().div(goal_heatmap.shape[0]).div(128)
            ll_goal_posterior = -F.l1_loss(goal_posterior, goal_heatmap, reduction='none').sum().div(goal_heatmap.shape[0]).div(128)

            # ll_traj_prior = torch.stack(ll_traj_prior).sum().div(batch)
            # ll_traj_prior = torch.tensor(0)
            loss_goal_pos = ll_traj_prior = (((pred_goal_pos_posterior - fut_traj[-1,:,:2])**2).mean() + loss_goal_pos_onehot).div(21)
            ll_traj_posterior = fut_rel_pos_dist_posterior.log_prob(fut_traj[:, :, 2:4]).sum().div(batch)

            loss_kl_goal =  torch.clamp(kl_divergence(q_dist_goal, p_dist_goal).sum().div(batch), min=0.07)
            loss_kl_goal_prior =  torch.clamp(kl_divergence(q_dist_goal, unif_prior_goal).sum().div(batch), min=0.07)

            loss_kl = torch.clamp(kl_divergence(q_dist, p_dist).sum().div(batch), min=0.07)

            goal_elbo = (ll_goal_prior + ll_goal_posterior).div(21) -  self.kl_weight_goal * (loss_kl_goal + loss_kl_goal_prior)
            traj_elbo = ll_traj_posterior - self.kl_weight * loss_kl

            #### map AE ####
            recon_mapX, pred_velX = self.map_decoder(map_featX)
            recon_mapY, pred_velY = self.map_decoder(map_featY)
            # past
            obs_obst = obs_obst.reshape(-1, obs_obst.shape[2], obs_obst.shape[3], obs_obst.shape[4])
            recon_map_lossX = - (torch.log(recon_mapX + self.eps) *  obs_obst +
                      torch.log(1 - recon_mapX + self.eps) * (1 - obs_obst)).sum().div(recon_mapX.shape[0])
            recon_velX = F.mse_loss(pred_velX, obs_traj[:,:,2:4].reshape(-1,2), reduction='sum').div(recon_mapX.shape[0])
            # fut
            fut_obst = fut_obst.reshape(-1, fut_obst.shape[2], fut_obst.shape[3], fut_obst.shape[4])
            recon_map_lossY = - (torch.log(recon_mapY + self.eps) *  fut_obst +
                      torch.log(1 - recon_mapY + self.eps) * (1 - fut_obst)).sum().div(recon_mapY.shape[0])
            recon_velY = F.mse_loss(pred_velY, fut_traj[:,:,2:4].reshape(-1,2), reduction='sum').div(recon_mapY.shape[0])

            loss_map = recon_map_lossX + recon_map_lossY
            loss_vel = recon_velX + recon_velY


            ######## Total Loss ######
            loss = - self.goal_vae_w * goal_elbo - traj_elbo + loss_map + loss_vel + loss_goal_pos



            self.optim_vae.zero_grad()
            loss.backward()
            self.optim_vae.step()


            # save model parameters
            if iteration % self.ckpt_save_iter == 0:
                self.save_checkpoint(iteration)


            # (visdom) insert current line stats
            if self.viz_on and (iteration % self.viz_ll_iter == 0):
                ade_min, fde_min, \
                ade_avg, fde_avg, \
                ade_std, fde_std, \
                test_loss_recon, test_loss_kl, test_loss, test_loss_map, test_loss_vel = self.evaluate_dist(self.val_loader, 20, loss=True)
                self.line_gather.insert(iter=iteration,
                                        loss_recon=-ll_traj_posterior.item(),
                                        loss_recon_prior=-ll_traj_prior.item(),
                                        loss_recon_goal=-ll_goal_posterior.item(),
                                        loss_recon_goal_prior=-ll_goal_prior.item(),
                                        loss_kl=loss_kl.item(),
                                        loss_kl_goal=loss_kl_goal.item(),
                                        loss_kl_goal_prior=loss_kl_goal_prior.item(),
                                        loss_map=loss_map.item(),
                                        loss_vel=loss_vel.item(),
                                        total_loss=-traj_elbo.item(),
                                        ade_min=ade_min,
                                        fde_min=fde_min,
                                        ade_avg=ade_avg,
                                        fde_avg=fde_avg,
                                        ade_std=ade_std,
                                        fde_std=fde_std,
                                        test_loss_recon=-test_loss_recon.item(),
                                        test_loss_kl=test_loss_kl.item(),
                                        test_total_loss=test_loss.item(),
                                        test_loss_map=test_loss_map.item(),
                                        test_loss_vel=test_loss_vel.item()
                                        )
                prn_str = ('[iter_%d (epoch_%d)] vae_loss: %.3f ' + \
                              '(recon: %.3f, kl: %.3f)\n' + \
                              'ADE min: %.2f, FDE min: %.2f, ADE avg: %.2f, FDE avg: %.2f\n'
                          ) % \
                          (iteration, epoch,
                           -traj_elbo.item(), -ll_traj_posterior.item(), loss_kl.item(),
                           ade_min, fde_min, ade_avg, fde_avg
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


    def evaluate_helper(self, error, seq_start_end):
        sum_min = 0
        sum_avg = 0
        sum_std = 0
        error = torch.stack(error, dim=1)

        for (start, end) in seq_start_end:
            start = start.item()
            end = end.item()
            _error = error[start:end]
            _error = torch.sum(_error, dim=0)
            sum_min += torch.min(_error)
            sum_avg += torch.mean(_error)
            sum_std += torch.std(_error)
        return sum_min, sum_avg, sum_std


    def evaluate_helper2(self, error, seq_start_end):
        sum_min = []
        sum_avg = []
        sum_std = []
        error = torch.stack(error, dim=1)

        for (start, end) in seq_start_end:
            start = start.item()
            end = end.item()
            _error = error[start:end]
            _error = torch.sum(_error, dim=0)
            sum_min.append(torch.min(_error).item()/(end-start))
            sum_avg.append(torch.mean(_error).item()/(end-start))
            sum_std.append(torch.std(_error).item()/(end-start))
        return np.concatenate([np.stack([sum_min, sum_avg, sum_std]).transpose(1,0), seq_start_end.cpu().numpy()], axis=1)


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






    def evaluate_dist(self, data_loader, num_samples, loss=False):
        self.set_mode(train=False)
        total_traj = 0

        loss_recon = loss_kl = total_loss = 0
        loss_map_recon = loss_vel = 0

        all_ade =[]
        all_fde =[]
        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
                 obs_frames, fut_frames, obs_obst, fut_obst, map_path, inv_h_t, obs_heatmap, goal_heatmap) = batch
                batch_size = obs_traj.size(1)
                total_traj += fut_traj.size(1)

                #######################
                #---
                # Traj Enc
                (encX_h_feat, logitX, map_featX) \
                    = self.encoderMx(obs_traj_st, seq_start_end, obs_obst, obs_traj[:,:,2:4])
                p_dist = discrete(logits=logitX)
                relaxed_p_dist = concrete(logits=logitX, temperature=self.temp)

                # Goal Enc
                goal_encX_h_feat, logitX_goal = self.encoderMx_goal(obs_heatmap, train=False)
                relaxed_p_dist_goal = concrete(logits=logitX_goal, temperature=self.goal_temp)


                if loss:
                    (encY_h_feat, logitY, map_featY) \
                        = self.encoderMy(obs_traj_st[-1], fut_vel_st, seq_start_end, encX_h_feat, fut_obst, fut_traj[:,:,2:4])

                    goal_prior = self.decoderM_goal(goal_encX_h_feat, relaxed_p_dist_goal.rsample())

                    recon_goal_heatmap = F.interpolate(goal_prior, size=(224, 224))
                    # recon_goal_heatmap2 = F.interpolate(goal_heatmap, size=(224,224))
                    pred_goal_pos_prior = []
                    for idx, (start, end) in enumerate(seq_start_end):
                        for t in range(start, end):
                            x_goal_pixel = 0
                            y_goal_pixel = 0
                            exp_recon = torch.exp(recon_goal_heatmap[t, 0] * (30 / recon_goal_heatmap[t].max()))
                            for pixel_idx in range(224):
                                x_goal_pixel += pixel_idx * exp_recon[pixel_idx, :].sum() / exp_recon.sum()
                                y_goal_pixel += pixel_idx * exp_recon[:, pixel_idx].sum() / exp_recon.sum()
                            goal_pixel = torch.cat(
                                [torch.round(y_goal_pixel).unsqueeze(0), torch.round(x_goal_pixel).unsqueeze(0),
                                 torch.ones(1)])
                            pred_goal = torch.matmul(goal_pixel, torch.tensor(np.linalg.inv(inv_h_t[idx])).float())
                            pred_goal_pos_prior.append(pred_goal / pred_goal[2])
                    pred_goal_pos_prior = torch.stack(pred_goal_pos_prior)[:, :2]


                    q_dist = discrete(logits=logitY)
                    fut_rel_pos_dist = self.decoderMy(
                        obs_traj_st[-1],
                        encX_h_feat,
                        relaxed_p_dist.rsample(), pred_goal_pos_prior,
                        obs_obst[-1].unsqueeze(0),
                        map_info=[seq_start_end, map_path, inv_h_t, lambda x: integrate_samples(x, obs_traj[-1, :, :2], dt=self.dt)]
                    )

                    recon_mapX, pred_velX = self.map_decoder(map_featX)
                    recon_mapY, pred_velY = self.map_decoder(map_featY)
                    ################## total loss for vae ####################
                    loglikelihood = fut_rel_pos_dist.log_prob(fut_traj[:, :, 2:4]).sum().div(batch_size)

                    kld = kl_divergence(q_dist, p_dist).sum().div(batch_size)
                    kld = torch.clamp(kld, min=0.07)
                    elbo = loglikelihood - self.kl_weight * kld
                    loss_recon +=loglikelihood
                    loss_kl +=kld

                    ## map AE
                    reshaped_obs_obst = obs_obst.reshape(-1, obs_obst.shape[2], obs_obst.shape[3], obs_obst.shape[4])
                    recon_map_lossX = - (torch.log(recon_mapX + self.eps) * reshaped_obs_obst +
                                         torch.log(1 - recon_mapX + self.eps) * (1 - reshaped_obs_obst)).sum().div(
                        recon_mapX.shape[0])
                    recon_velX = F.mse_loss(pred_velX, obs_traj[:, :, 2:4].reshape(-1, 2), reduction='sum').div(
                        recon_mapX.shape[0])

                    ## map AE
                    reshaped_fut_obst = fut_obst.reshape(-1, fut_obst.shape[2], fut_obst.shape[3], fut_obst.shape[4])
                    recon_map_lossY = - (torch.log(recon_mapY + self.eps) * reshaped_fut_obst +
                                         torch.log(1 - recon_mapY + self.eps) * (1 - reshaped_fut_obst)).sum().div(
                        recon_mapY.shape[0])
                    recon_velY = F.mse_loss(pred_velY, fut_traj[:, :, 2:4].reshape(-1, 2), reduction='sum').div(
                        recon_mapY.shape[0])

                    total_loss += (-elbo + (recon_map_lossX + recon_velX) + (recon_map_lossY + recon_velY))
                    # loss_map_recon += (recon_map_lossX + recon_map_lossY)
                    # loss_vel += (recon_velX + recon_velY)
                    loss_vel +=((pred_goal_pos_prior - fut_traj[-1,:,:2])**2).mean()

                ade, fde = [], []
                loss_goal_pos_onehot = 0
                for w_one_hot in torch.eye(20):
                    w_one_hot = w_one_hot.unsqueeze(0).repeat((goal_encX_h_feat.shape[0], 1)).to(self.device)

                    goal_prior = self.decoderM_goal(goal_encX_h_feat, w_one_hot)
                    recon_goal_heatmap = F.interpolate(goal_prior, size=(224, 224))
                    pred_goal_pos_prior = []
                    for idx, (start, end) in enumerate(seq_start_end):
                        for t in range(start, end):
                            x_goal_pixel = 0
                            y_goal_pixel = 0
                            exp_recon = torch.exp(recon_goal_heatmap[t, 0] * (30 / recon_goal_heatmap[t].max()))
                            for pixel_idx in range(224):
                                x_goal_pixel += pixel_idx * exp_recon[pixel_idx, :].sum() / exp_recon.sum()
                                y_goal_pixel += pixel_idx * exp_recon[:, pixel_idx].sum() / exp_recon.sum()
                            goal_pixel = torch.cat(
                                [torch.round(y_goal_pixel).unsqueeze(0), torch.round(x_goal_pixel).unsqueeze(0),
                                 torch.ones(1)])
                            pred_goal = torch.matmul(goal_pixel, torch.tensor(np.linalg.inv(inv_h_t[idx])).float())
                            pred_goal_pos_prior.append(pred_goal / pred_goal[2])
                    pred_goal_pos_prior = torch.stack(pred_goal_pos_prior)[:, :2]
                    loss_goal_pos_onehot += ((pred_goal_pos_prior - fut_traj[-1, :, :2]) ** 2).mean()


                    fut_rel_pos_dist = self.decoderMy(
                        obs_traj_st[-1],
                        encX_h_feat,
                        torch.cat([relaxed_p_dist.rsample(), w_one_hot], dim=1),
                        obs_obst[-1].unsqueeze(0),
                        map_info=[seq_start_end, map_path, inv_h_t, lambda x: integrate_samples(x, obs_traj[-1, :, :2], dt=self.dt)]
                    )

                    pred_fut_traj_rel = fut_rel_pos_dist.rsample()
                    pred_fut_traj=integrate_samples(pred_fut_traj_rel, obs_traj[-1, :, :2], dt=self.dt)

                    ade.append(displacement_error(
                        pred_fut_traj, fut_traj[:,:,:2], mode='raw'
                    ))
                    fde.append(final_displacement_error(
                        pred_fut_traj[-1], fut_traj[-1,:,:2], mode='raw'
                    ))
                if loss:
                    loss_map_recon += loss_goal_pos_onehot.div(20)

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
                   loss_recon/b, loss_kl/b, total_loss/b, loss_map_recon/b, loss_vel/b
        else:
            return ade_min, fde_min, \
                   ade_avg, fde_avg, \
                   ade_std, fde_std





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
                (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
                 obs_frames, pred_frames, obs_obst, fut_obst, map_path, inv_h_t) = batch
                total_traj += fut_traj.size(1)

                # rng = range(9, 18)
                # rng = range(18, 38)
                # rng = range(38, 45)

                # rng = range(17)
                # rng = range(95, 104)
                # rng = range(100, 101)

                # rng = range(19,27)

                # Note: returns angle in radians
                def theta(v, w):
                    return np.arccos(v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w)))


                rng = range(30,31)
                rng = range(180,181)
                map = imageio.imread(map_path[rng[0]])
                fig, ax = plt.subplots()
                # ax.imshow(map)
                for idx in rng:
                    obs_real = obs_traj[:, idx, :2]
                    obs_real = np.concatenate([obs_real, np.ones((self.obs_len, 1))], axis=1)
                    obs_pixel = np.matmul(obs_real, inv_h_t[idx])
                    obs_pixel /= np.expand_dims(obs_pixel[:, 2], 1)

                    gt_real = fut_traj[:, idx, :2]
                    gt_real = np.concatenate([gt_real, np.ones((self.pred_len, 1))], axis=1)
                    gt_pixel = np.matmul(gt_real, inv_h_t[idx])
                    gt_pixel /= np.expand_dims(gt_pixel[:, 2], 1)
                    gt_data = np.concatenate([obs_pixel, gt_pixel], 0)[:,:2]
                    gt_data[:, [0, 1]] = gt_data[:, [1, 0]]

                    per_step_dist = (((obs_pixel[1:, :2] - obs_pixel[:-1, :2]) ** 2).sum(1) ** (1 / 2)).mean()
                    circle = np.zeros((112,112))
                    for x in range(112):
                        for y in range(112):
                            # dist = np.linalg.norm([x,y] - gt_data[7]) - per_step_dist*12
                            # if dist < 1 and dist > -1:
                            dist_from_last_obs = np.linalg.norm([x, y] - gt_data[7])
                            if dist_from_last_obs < per_step_dist * (self.pred_len+1):
                                angle = theta(([x, y] - (gt_data[7] - gt_data[6])) - gt_data[6],
                                              gt_data[7] - gt_data[6])
                                # circle.append([x,y,np.cos(angle)])
                                if np.cos(angle) >= 0:
                                    # circle.append([x, y])
                                    circle[x, y] = np.cos(angle) * (1+dist_from_last_obs) + 1 # in case dist_from_last_obs < 1
                                    # map[x, y] =  circle[x, y]


                    # fig, ax = plt.subplots()
                    # ax.imshow(map)
                    # for x in range(112):
                    #     for y in range(112):
                    #         if circle[x,y] > 0:
                    #             ax.scatter(y, x, s=2, color='orange', alpha=0.6)# 'hotpink', '#88c999'

                    # print only circle
                    fig, ax = plt.subplots()
                    ax.imshow(circle * map)
                    ax.scatter(gt_data[:8,1], gt_data[:8,0], s=5, c='b', marker='x')
                    ax.scatter(gt_data[8:20,1], gt_data[8:20,0], s=5, c='r', marker='x')
                    # ax.scatter(gt_data[-1,0], gt_data[-1,1], s=5)


                    candidate_pos_ic = np.array(np.where(circle * map > 0)).transpose((1,0))

                    ### random selection ###
                    # random_idx = list(range(candidate_pos_ic.shape[0]))
                    # np.random.shuffle(random_idx)
                    # selected_goal_ic = candidate_pos_ic[random_idx[:19]]
                    # for coord in selected_goal_ic:
                    #     ax.scatter(coord[1], coord[0], s=5, c='hotpink', marker='^')


                    ### highest 19 prob ###
                    # prob = (circle * map)[candidate_pos_ic[:,0],candidate_pos_ic[:,1]]
                    # prob = np.exp(prob)/sum(np.exp(prob))
                    # random_idx_with_prob = np.random.choice(range(candidate_pos_ic.shape[0]), 19 , replace=False, p=prob)
                    # selected_goal_ic = candidate_pos_ic[random_idx_with_prob]
                    # for coord in selected_goal_ic:
                    #     ax.scatter(coord[1], coord[0], s=5, c='hotpink', marker='^')

                    ### reculrsive greed ###

                    radius = per_step_dist * (self.pred_len + 1) / 8

                    def find_coord(goal_map, updated_map, selected_goal_ic, candidate_pos_ic, radius):
                        goal_pos_idx = updated_map[candidate_pos_ic[:, 0], candidate_pos_ic[:, 1]].argmax()
                        selected_goal_ic.append(
                            [candidate_pos_ic[goal_pos_idx, 0], candidate_pos_ic[goal_pos_idx, 1]])
                        # print(selected_goal_ic)

                        if len(selected_goal_ic) == 19:
                            return selected_goal_ic
                        else:
                            # 반경의 모든 coord의 prob를 0으로.
                            updated_map[selected_goal_ic[-1][0], selected_goal_ic[-1][1]] = 0
                            for coord in candidate_pos_ic:
                                if np.linalg.norm(selected_goal_ic[-1] - coord) < radius:
                                    updated_map[coord[0], coord[1]] = 0
                                    # print(updated_map[coord])

                            if updated_map.sum() == 0:
                                # return selected_goal_ic
                                print('>>>> replay')
                                updated_map = goal_map.copy()
                                # 선택된 coord를 제외하고 다시 시작
                                for coord in selected_goal_ic:
                                    updated_map[coord[0], coord[1]] = 0
                            selected_goal_ic = find_coord(goal_map, updated_map, selected_goal_ic, candidate_pos_ic, radius)
                        return selected_goal_ic

                    selected_goal_ic = find_coord(circle * map, circle * map, [], candidate_pos_ic, radius)

                    for coord in selected_goal_ic:
                        ax.scatter(coord[1], coord[0], s=5, c='hotpink', marker='^')
                        # ax.scatter(coord[1], coord[0], s=5, c='black', marker='^')
                        # ax.scatter(coord[1], coord[0], s=5, c='#88c999', marker='X')

                from scipy import ndimage
                # downsample goal map
                vir_goal_map = np.zeros((112,112))
                for coord in selected_goal_ic:
                    vir_goal_map[coord[0], coord[1]] = 1
                # vir_goal_map = ndimage.filters.gaussian_filter(vir_goal_map, sigma=1)
                plt.imshow(vir_goal_map)

                vir_goal_map = transforms.Compose([
                    transforms.Resize(100),
                    transforms.ToTensor()
                ])(Image.fromarray(vir_goal_map))
                plt.imshow(vir_goal_map[0])

                # upsample goal map
                vir_goal_map = F.interpolate(vir_goal_map.unsqueeze(0), size=(112, 112))
                plt.imshow(vir_goal_map[0][0])

                selected_goal_ic = np.array(selected_goal_ic)
                selected_goal_ic[:, [0, 1]] = selected_goal_ic[:, [1, 0]]
                selected_goal_ic = np.concatenate([selected_goal_ic, np.ones((selected_goal_ic.shape[0],1))],1)

                # goal in wcs
                d= np.matmul(selected_goal_ic, np.linalg.inv(inv_h_t[idx]))
                back_real = d / np.expand_dims(d[:, 2], 1)
                plt.scatter(obs_traj[:, idx, 0], obs_traj[:, idx, 1], c='b')
                plt.scatter(fut_traj[:, idx, 0], fut_traj[:, idx, 1], c='r')
                plt.scatter(back_real[:,0], back_real[:,1], c='g')



                im = imageio.imread(map_path[rng[0]]) / 255
                np.expand_dims(im,-1)

                ## create heatmap
                from scipy import ndimage
                x = np.zeros((224, 224))
                x[74, 81] = 1
                heat_map = ndimage.filters.gaussian_filter(x, sigma=5)
                plt.imshow(heat_map)

                ## resize to 100
                im = Image.fromarray(heat_map)
                image = transforms.Compose([
                    transforms.Resize(100),
                    transforms.ToTensor()
                ])(im)
                plt.imshow(image[0])

                ## back to 224
                im2 = Image.fromarray(image.numpy()[0])
                image2 = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor()
                ])(im2)
                plt.imshow(image2[0])



    def plot_traj_var(self, data_loader, num_samples=20):
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
                (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
                 obs_frames, pred_frames, obs_obst, fut_obst, map_path, inv_h_t) = batch
                total_traj += fut_traj.size(1)

                # path = '../datasets\hotel\\test\\biwi_hotel.txt'
                # l=f.readlines()
                # data = read_file(path, 'tab')
                # framd_num=6980
                # np.where(obs_frames[:, 0] == framd_num)
                # d = data[1989:2000]
                # gt_real = d[..., -2:]
                # gt_real = np.concatenate([gt_real, np.ones((2000-1989, 1))], axis=1)
                # gt_pixel = np.matmul(gt_real, inv_h_t)
                # gt_pixel /= np.expand_dims(gt_pixel[:, 2], 1)
                #
                # fig, ax = plt.subplots()
                # cap.set(1, framd_num)
                # _, frame = cap.read()
                # ax.imshow(frame)
                # for i in range(len(d)):
                #     ax.text(gt_pixel[i][1], gt_pixel[i][0], str(int(d[:,1][i])), fontsize=10)



                (encX_h_feat, logitX, map_featX) \
                    = self.encoderMx(obs_traj_st, seq_start_end, obs_obst, obs_traj[:,:,2:4])
                relaxed_p_dist = concrete(logits=logitX, temperature=self.temp)

                # (s,e) = seq_start_end[j]
                # np.where(1658 in seq_start_end)

                for j, (s, e) in enumerate(seq_start_end):
                    agent_rng = range(s, e)
                    seq_map = imageio.imread(map_path[j])  # seq = 한 씬에서 모든 neighbors니까. 같은 데이터셋.

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

                        gt_data, pred_data = [], []

                        for idx in range(len(agent_rng)):
                            one_ped = agent_rng[idx]
                            obs_real = obs_traj[:, one_ped,:2]
                            obs_real = np.concatenate([obs_real, np.ones((self.obs_len, 1))], axis=1)
                            obs_pixel = np.matmul(obs_real, inv_h_t[j])
                            obs_pixel /= np.expand_dims(obs_pixel[:, 2], 1)

                            gt_real = fut_traj[:, one_ped, :2]
                            gt_real = np.concatenate([gt_real, np.ones((self.pred_len, 1))], axis=1)
                            gt_pixel = np.matmul(gt_real, inv_h_t[j])
                            gt_pixel /= np.expand_dims(gt_pixel[:, 2], 1)

                            pred_real = pred_fut_traj[:, one_ped].numpy()
                            pred_pixel = np.concatenate([pred_real, np.ones((self.pred_len, 1))], axis=1)
                            pred_pixel = np.matmul(pred_pixel, inv_h_t[j])
                            pred_pixel /= np.expand_dims(pred_pixel[:, 2], 1)

                            gt_data.append(np.concatenate([obs_pixel, gt_pixel], 0)) # (20, 3)
                            pred_data.append(np.concatenate([obs_pixel, pred_pixel], 0))

                        gt_data = np.stack(gt_data)
                        pred_data = np.stack(pred_data)

                        # if self.dataset_name == 'eth':
                        # gt_data[:,:, [0,1]] = gt_data[:,:,[1,0]]
                        # pred_data[:,:,[0,1]] = pred_data[:,:,[1,0]]

                        multi_sample_pred.append(pred_data)

                    def init():
                        ax.imshow(seq_map)

                    def update_dot(num_t):
                        print(num_t)
                        ax.imshow(seq_map)

                        for i in range(n_agent):
                            ln_gt[i].set_data(gt_data[i, :num_t, 0], gt_data[i, :num_t, 1])

                            for j in range(20):
                                all_ln_pred[i][j].set_data(multi_sample_pred[j][i, :num_t, 0],
                                                           multi_sample_pred[j][i, :num_t, 1])
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
                        ln_pred = []
                        for _ in range(20):
                            ln_pred.append(ax.plot([], [], colors[i % len(colors)], alpha=0.6, linewidth=1)[0])
                        all_ln_pred.append(ln_pred)


                    ani = FuncAnimation(fig, update_dot, frames=n_frame, interval=1, init_func=init())

                    # writer = PillowWriter(fps=3000)

                    ani.save(gif_path + "/" +self.dataset_name+ "_" + title + "_agent" + str(agent_rng[0]) +"to" +str(agent_rng[-1]) +".gif", fps=4)


    ####
    def viz_init(self):
        self.viz.close(env=self.name + '/lines', win=self.win_id['loss_recon'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['loss_kl'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['loss_map'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['loss_vel'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['total_loss'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['test_loss_recon'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['test_loss_kl'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['test_loss_map'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['test_loss_vel'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['test_total_loss'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['ade_min'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['fde_min'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['ade_avg'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['fde_avg'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['ade_std'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['fde_std'])

        self.viz.close(env=self.name + '/lines', win=self.win_id['loss_recon_prior'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['loss_recon_goal'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['loss_recon_goal_prior'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['loss_kl_goal'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['loss_kl_goal_prior'])


    ####
    def visualize_line(self):

        # prepare data to plot
        data = self.line_gather.data
        iters = torch.Tensor(data['iter'])
        loss_recon = torch.Tensor(data['loss_recon'])
        loss_kl = torch.Tensor(data['loss_kl'])
        loss_map = torch.Tensor(data['loss_map'])
        loss_vel = torch.Tensor(data['loss_vel'])
        total_loss = torch.Tensor(data['total_loss'])
        ade_min = torch.Tensor(data['ade_min'])
        fde_min = torch.Tensor(data['fde_min'])
        ade_avg = torch.Tensor(data['ade_avg'])
        fde_avg = torch.Tensor(data['fde_avg'])
        ade_std = torch.Tensor(data['ade_std'])
        fde_std = torch.Tensor(data['fde_std'])
        test_loss_recon = torch.Tensor(data['test_loss_recon'])
        test_loss_kl = torch.Tensor(data['test_loss_kl'])
        test_total_loss = torch.Tensor(data['test_total_loss'])
        test_loss_map = torch.Tensor(data['test_loss_map'])
        test_loss_vel = torch.Tensor(data['test_loss_vel'])

        loss_recon_prior = torch.Tensor(data['loss_recon_prior'])
        loss_recon_goal = torch.Tensor(data['loss_recon_goal'])
        loss_recon_goal_prior = torch.Tensor(data['loss_recon_goal_prior'])
        loss_kl_goal = torch.Tensor(data['loss_kl_goal'])
        loss_kl_goal_prior = torch.Tensor(data['loss_kl_goal_prior'])

        self.viz.line(
            X=iters, Y=loss_recon_prior, env=self.name + '/lines',
            win=self.win_id['loss_recon_prior'], update='append',
            opts=dict(xlabel='iter', ylabel='loss_recon_prior',
                      title='loss_recon_prior')
        )

        self.viz.line(
            X=iters, Y=loss_recon_goal, env=self.name + '/lines',
            win=self.win_id['loss_recon_goal'], update='append',
            opts=dict(xlabel='iter', ylabel='loss_recon_goal',
                      title='loss_recon_goal')
        )

        self.viz.line(
            X=iters, Y=loss_kl_goal_prior, env=self.name + '/lines',
            win=self.win_id['loss_kl_goal_prior'], update='append',
            opts=dict(xlabel='iter', ylabel='loss_kl_goal_prior',
                      title='loss_kl_goal_prior')
        )

        self.viz.line(
            X=iters, Y=loss_kl_goal, env=self.name + '/lines',
            win=self.win_id['loss_kl_goal'], update='append',
            opts=dict(xlabel='iter', ylabel='loss_kl_goal',
                      title='loss_kl_goal')
        )

        self.viz.line(
            X=iters, Y=loss_recon_goal_prior, env=self.name + '/lines',
            win=self.win_id['loss_recon_goal_prior'], update='append',
            opts=dict(xlabel='iter', ylabel='loss_recon_goal_prior',
                      title='loss_recon_goal_prior')
        )


        self.viz.line(
            X=iters, Y=loss_recon, env=self.name + '/lines',
            win=self.win_id['loss_recon'], update='append',
            opts=dict(xlabel='iter', ylabel='-loglikelihood',
                      title='Recon. loss of predicted future traj')
        )

        self.viz.line(
            X=iters, Y=loss_kl, env=self.name + '/lines',
            win=self.win_id['loss_kl'], update='append',
            opts=dict(xlabel='iter', ylabel='kl divergence',
                      title='KL div. btw posterior and c. prior'),
        )

        self.viz.line(
            X=iters, Y=loss_map, env=self.name + '/lines',
            win=self.win_id['loss_map'], update='append',
            opts=dict(xlabel='iter', ylabel='loss',
                      title='Map recon loss'),
        )

        self.viz.line(
            X=iters, Y=loss_vel, env=self.name + '/lines',
            win=self.win_id['loss_vel'], update='append',
            opts=dict(xlabel='iter', ylabel='loss',
                      title='Velocity loss'),
        )

        self.viz.line(
            X=iters, Y=total_loss, env=self.name + '/lines',
            win=self.win_id['total_loss'], update='append',
            opts=dict(xlabel='iter', ylabel='total loss',
                      title='Total loss'),
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
            X=iters, Y=test_total_loss, env=self.name + '/lines',
            win=self.win_id['test_total_loss'], update='append',
            opts=dict(xlabel='iter', ylabel='total loss',
                      title='Test Total loss'),
        )

        self.viz.line(
            X=iters, Y=test_loss_map, env=self.name + '/lines',
            win=self.win_id['test_loss_map'], update='append',
            opts=dict(xlabel='iter', ylabel='loss',
                      title='Test MAP recon loss'),
        )

        self.viz.line(
            X=iters, Y=test_loss_vel, env=self.name + '/lines',
            win=self.win_id['test_loss_vel'], update='append',
            opts=dict(xlabel='iter', ylabel='loss',
                      title='Test Velocity loss'),
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

            self.map_encoder.train()
            self.map_decoder.train()
            self.encoderMx_goal.train()
            self.encoderM_goal.train()
            self.decoderM_goal.train()

        else:
            self.encoderMx.eval()
            self.encoderMy.eval()
            self.decoderMy.eval()

            self.map_encoder.eval()
            self.map_decoder.eval()
            self.encoderMx_goal.eval()
            self.encoderM_goal.eval()
            self.decoderM_goal.eval()

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
        map_dec_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_map_dec.pt' % iteration
        )
        map_enc_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_map_enc.pt' % iteration
        )

        encoderMx_goal_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderMx_goal.pt' % iteration
        )
        encoderM_goal_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderM_goal.pt' % iteration
        )
        decoderM_goal_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_decoderM_goal.pt' % iteration
        )

        mkdirs(self.ckpt_dir)

        torch.save(self.encoderMx, encoderMx_path)
        torch.save(self.encoderMy, encoderMy_path)
        torch.save(self.decoderMy, decoderMy_path)
        torch.save(self.map_decoder, map_dec_path)
        torch.save(self.map_encoder, map_enc_path)

        torch.save(self.encoderMx_goal, encoderMx_goal_path)
        torch.save(self.encoderM_goal, encoderM_goal_path)
        torch.save(self.decoderM_goal, decoderM_goal_path)
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

        map_encoder_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_map_enc.pt' % self.ckpt_load_iter
        )

        map_decoder_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_map_dec.pt' % self.ckpt_load_iter
        )

        encoderMx_goal_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderMx_goal.pt' % self.ckpt_load_iter
        )
        encoderM_goal_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderM_goal.pt' % self.ckpt_load_iter
        )
        decoderM_goal_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_decoderM_goal.pt' % self.ckpt_load_iter
        )


        if self.device == 'cuda':
            self.encoderMx = torch.load(encoderMx_path)
            self.encoderMy = torch.load(encoderMy_path)
            self.decoderMy = torch.load(decoderMy_path)
            self.map_encoder = torch.load(map_encoder_path)
            self.map_decoder = torch.load(map_decoder_path)

            self.encoderMx_goal = torch.load(encoderMx_goal_path)
            self.encoderM_goal = torch.load(encoderM_goal_path)
            self.decoderM_goal = torch.load(decoderM_goal_path)
        else:
            self.encoderMx = torch.load(encoderMx_path, map_location='cpu')
            self.encoderMy = torch.load(encoderMy_path, map_location='cpu')
            self.decoderMy = torch.load(decoderMy_path, map_location='cpu')
            self.map_encoder = torch.load(map_encoder_path, map_location='cpu')
            self.map_decoder = torch.load(map_decoder_path, map_location='cpu')

            self.encoderMx_goal = torch.load(torch.load(encoderMx_goal_path), map_location='cpu')
            self.encoderM_goal = torch.load(torch.load(encoderM_goal_path), map_location='cpu')
            self.decoderM_goal = torch.load(torch.load(decoderM_goal_path), map_location='cpu')


    def load_map_encoder(self):
        map_encoder_path = 'ckpts/A2E_map_size_16_drop_out0.1_hidden_d256_latent_d32_run_4/iter_20000_encoder.pt'
        map_decoder_path = 'ckpts/A2E_map_size_16_drop_out0.1_hidden_d256_latent_d32_run_4/iter_20000_decoder.pt'
        if self.device == 'cuda':
            self.map_encoder = torch.load(map_encoder_path)
            self.map_decoder = torch.load(map_decoder_path)
        else:
            self.map_encoder = torch.load(map_encoder_path, map_location='cpu')
            self.map_decoder = torch.load(map_decoder_path, map_location='cpu')
