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
from unet.probabilistic_unet import ProbabilisticUnet
from unet.unet import Unet
import numpy as np
import visdom


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

        self.name = '%s_pred_len_%s_zD_%s_wD_%s_dr_mlp_%s_dr_rnn_%s_enc_hD_%s_dec_hD_%s_mlpD_%s_map_featD_%s_map_mlpD_%s_lr_%s_klw_%s_lg_klw_%s_ll_prior_w_%s' % \
                    (args.dataset_name, args.pred_len, args.zS_dim, args.w_dim, args.dropout_mlp, args.dropout_rnn, args.encoder_h_dim,
                     args.decoder_h_dim, args.mlp_dim, args.map_feat_dim , args.map_mlp_dim, args.lr_VAE, args.kl_weight, args.lg_kl_weight, args.ll_prior_w)


        # to be appended by run_id

        # self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = args.device
        self.temp=1.99
        self.dt=0.4
        self.eps=1e-9
        self.ll_prior_w =args.ll_prior_w
        self.sg_idx =  np.array([3,7,11])

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
                lg_recon='win_lg_recon', sg_recon='win_sg_recon', lg_kl='win_lg_kl',
                test_lg_recon='win_test_lg_recon', test_sg_recon='win_test_sg_recon', test_lg_kl='win_test_lg_kl',
                sg_ade_min='win_sg_ade_min', sg_ade_avg='win_sg_ade_avg', sg_ade_std='win_sg_ade_std'
            )
            self.line_gather = DataGather(
                'iter', 'loss_recon', 'loss_kl',  'loss_recon_prior',
                'ade_min', 'fde_min', 'ade_avg', 'fde_avg', 'ade_std', 'fde_std',
                'test_loss_recon', 'test_loss_kl',
                'lg_recon', 'sg_recon', 'lg_kl',
                'test_lg_recon', 'test_sg_recon', 'test_lg_kl',
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
            # self.encoderLG = LGEncoder(
            #     args.zS_dim,
            #     mlp_dim=args.mlp_dim,
            #     drop_out_conv=args.dropout_rnn,
            #     drop_out_mlp=args.dropout_mlp,
            #     device=self.device).to(self.device)

            # input = env + 8 past / output = env + lg
            num_filters = [32,32,64,64,64,128]
            self.lg_cvae = ProbabilisticUnet(input_channels=9, num_classes=2, num_filters=num_filters, latent_dim=self.w_dim,
                                    no_convs_fcomb=4, beta=self.lg_kl_weight).to(self.device)

            # input = env + 8 past + lg / output = env + sg(including lg)
            self.sg_unet = Unet(input_channels=10, num_classes=4, num_filters=num_filters,
                             apply_last_layer=True, padding=True).to(self.device)


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
                mlp_dim=args.mlp_dim,
                z_dim=args.zS_dim,
                num_layers=args.num_layers,
                device=args.device,
                dropout_rnn=args.dropout_rnn).to(self.device)

        else:  # load a previously saved model
            print('Loading saved models (iter: %d)...' % self.ckpt_load_iter)
            self.load_checkpoint()
            print('...done')


        # get VAE parameters
        vae_params = \
            list(self.encoderMx.parameters()) + \
            list(self.encoderMy.parameters()) + \
            list(self.decoderMy.parameters()) + \
            list(self.lg_cvae.parameters()) + \
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

        self.recon_loss_with_logit = nn.BCEWithLogitsLoss(size_average = False, reduce=False, reduction=None)


    def make_heatmap(self, local_ic, local_map):
        obs_heat_map = []
        fut_heat_map = []
        for i in range(len(local_ic)):
            ohm = [local_map[i, 0].detach().cpu().numpy()]
            fhm = [local_map[i, 0].detach().cpu().numpy()]
            for t in range(self.obs_len + self.pred_len):
                # heat_map_traj = np.zeros_like(local_map[i, 0])
                heat_map_traj = np.zeros((160,160))
                heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 1
                heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj,
                                                                sigma=2)  # as Y-net used variance 4 for the GT heatmap representation.
                if t < self.obs_len:
                    ohm.append(heat_map_traj)
                else:
                    fhm.append(heat_map_traj)
            obs_heat_map.append(np.stack(ohm))
            fut_heat_map.append(np.stack(fhm))
        obs_heat_map = torch.tensor(np.stack(obs_heat_map)).float().to(self.device)
        fut_heat_map = np.stack(fut_heat_map)
        return obs_heat_map, fut_heat_map

    ####
    def train(self):
        self.set_mode(train=True)
        torch.autograd.set_detect_anomaly(True)
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


            (obs_traj, fut_traj, seq_start_end,
             obs_frames, pred_frames, map_path, inv_h_t,
             local_map, local_ic, local_homo) = next(iterator)
            batch_size = obs_traj.size(1) #=sum(seq_start_end[:,1] - seq_start_end[:,0])

###########
            obs_heat_map, fut_heat_map =  self.make_heatmap(local_ic, local_map)
            lg_heat_map = torch.tensor(fut_heat_map[:,[0,12]]).float().to(self.device)
            sg_heat_map = torch.tensor(fut_heat_map[:, np.concatenate([[0], self.sg_idx + 1])]).float().to(self.device)

            # idx=0
            # heat_map_traj = np.zeros_like(local_map[idx, 0])
            # heat_map_traj = local_map[idx, 0]
            # for t in range(20):
            #     heat_map_traj[local_ic[idx, t, 0], local_ic[idx, t, 1]] = 1
            # heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
            # plt.imshow(heat_map_traj)
            # plt.scatter(local_ic[idx, :8, 1], local_ic[idx, :8, 0])
            # plt.imshow(local_map[idx, 0])

            #-------- long term goal --------
            self.lg_cvae.forward(obs_heat_map, lg_heat_map, training=True)
            recon_lg_heat = self.lg_cvae.reconstruct(use_posterior_mean=False,
                                                   calculate_posterior=True)
            lg_kl = self.lg_cvae.kl_divergence(analytic=True).sum().div(batch_size)
            lg_recon_loss = self.recon_loss_with_logit(input=recon_lg_heat, target=lg_heat_map).sum().div(np.prod([*lg_heat_map.size()[:3]]))
            lg_elbo = -lg_recon_loss - self.lg_kl_weight * lg_kl


            #-------- short term goal --------
            # obs_lg_heat = torch.cat([obs_heat_map, lg_heat_map[:,-1].unsqueeze(1)], dim=1)
            recon_sg_heat = self.sg_unet.forward(torch.cat([obs_heat_map, lg_heat_map[:,-1].unsqueeze(1)], dim=1), training=True)
            sg_recon_loss = self.recon_loss_with_logit(input=recon_sg_heat, target=sg_heat_map).sum().div(np.prod([*sg_heat_map.size()[:3]]))


            #-------- trajectories --------
            (hx, mux, log_varx) \
                = self.encoderMx(obs_traj, seq_start_end, self.sg_unet.enc_feat, local_homo)

            # all_pixel_local = local_ic[0,:8]
            # h = local_homo[0]
            # back_wc = np.matmul(np.concatenate([all_pixel_local, np.ones((len(all_pixel_local), 1))], axis=1), np.transpose(h))
            # back_wc /= np.expand_dims(back_wc[:, 2], 1)
            # back_wc = back_wc[:,:2]
            # diff = back_wc-obs_traj[:,0,:2]

            (muy, log_vary) \
                = self.encoderMy(obs_traj[-1], fut_traj[:,:,2:4], seq_start_end, hx, train=True)

            p_dist = Normal(mux, torch.sqrt(torch.exp(log_varx)))
            q_dist = Normal(muy, torch.sqrt(torch.exp(log_vary)))


            # TF, goals, z~posterior
            fut_rel_pos_dist_tf_post = self.decoderMy(
                obs_traj[-1],
                hx,
                q_dist.rsample(),
                fut_traj[self.sg_idx, :, :2].permute(1,0,2), # goal
                self.sg_idx - 3,
                fut_traj # TF
            )

            pred_sg_wc = []
            for i in range(batch_size):
                # pred_sg_ic = []
                # for heat_map in sg_heat_map[i, 1:]:
                #     pred_sg_ic.append((heat_map == torch.max(heat_map)).nonzero()[0])
                # pred_sg_ic = torch.stack(pred_sg_ic)

                ## soft argmax
                pred_sg_ic = []
                for heat_map in sg_heat_map[i, 1:]:
                    x_goal_pixel = 0
                    y_goal_pixel = 0
                    exp_recon = torch.exp(heat_map * (20 / heat_map.max()))
                    # exp_recon = torch.exp(heat_map)
                    for pixel_idx in range(len(heat_map)):
                        x_goal_pixel += pixel_idx * exp_recon[pixel_idx, :].sum() / exp_recon.sum()
                        y_goal_pixel += pixel_idx * exp_recon[:, pixel_idx].sum() / exp_recon.sum()
                    pred_sg_ic.append(torch.cat(
                        [torch.round(x_goal_pixel).unsqueeze(0), torch.round(y_goal_pixel).unsqueeze(0)]
                    ))
                pred_sg_ic = torch.stack(pred_sg_ic)
                # ((local_ic[0,[11,15,19]] - pred_sg_ic) ** 2).sum(1).mean()
                back_wc = torch.matmul(
                    torch.cat([pred_sg_ic, torch.ones((len(pred_sg_ic), 1)).to(self.device)], dim=1),
                    torch.transpose(local_homo[i], 1, 0))
                back_wc /= back_wc[:, 2].unsqueeze(1)
                pred_sg_wc.append(back_wc[:,:2])
                # ((back_wc - fut_traj[[3, 7, 11], 0, :2]) ** 2).sum(1).mean()
            pred_sg_wc = torch.stack(pred_sg_wc)

            # NO TF, predicted goals, z~prior
            fut_rel_pos_dist_prior = self.decoderMy(
                obs_traj[-1],
                hx,
                p_dist.rsample(),
                pred_sg_wc, # goal
                self.sg_idx - 3
            )


            ll_tf_post = fut_rel_pos_dist_tf_post.log_prob(fut_traj[:, :, 2:4]).sum().div(batch_size)
            ll_prior = fut_rel_pos_dist_prior.log_prob(fut_traj[:, :, 2:4]).sum().div(batch_size)

            loss_kl = kl_divergence(q_dist, p_dist).sum().div(batch_size)
            loss_kl = torch.clamp(loss_kl, min=0.07)
            # print('log_likelihood:', loglikelihood.item(), ' kl:', loss_kl.item())

            loglikelihood= ll_tf_post + self.ll_prior_w * ll_prior
            traj_elbo = loglikelihood - self.kl_weight * loss_kl

            loss = - traj_elbo - lg_elbo + sg_recon_loss

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
                sg_ade_min, sg_ade_avg, sg_ade_std, \
                test_loss_recon, test_loss_kl,\
                test_lg_recon, test_lg_kl, test_sg_recon = self.evaluate_dist(self.val_loader, loss=True)
                self.line_gather.insert(iter=iteration,
                                        ade_min=ade_min,
                                        fde_min=fde_min,
                                        ade_avg=ade_avg,
                                        fde_avg=fde_avg,
                                        ade_std=ade_std,
                                        fde_std=fde_std,
                                        sg_ade_min=sg_ade_min,
                                        sg_ade_avg=sg_ade_avg,
                                        sg_ade_std=sg_ade_std,
                                        loss_recon=-ll_tf_post.item(),
                                        loss_recon_prior=-ll_prior.item(),
                                        loss_kl=loss_kl.item(),
                                        test_loss_recon=-test_loss_recon.item(),
                                        test_loss_kl=test_loss_kl.item(),
                                        lg_recon=lg_recon_loss.item(),
                                        lg_kl=lg_kl.item(),
                                        sg_recon=sg_recon_loss.item(),
                                        test_lg_recon=test_lg_recon.item(),
                                        test_lg_kl=test_lg_kl.item(),
                                        test_sg_recon=test_sg_recon.item()
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
                if self.record_file:
                    record = open(self.record_file, 'a')
                    record.write('%s\n' % (prn_str,))
                    record.close()


            # (visdom) visualize line stats (then flush out)
            if self.viz_on and (iteration % self.viz_la_iter == 0):
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
        lg_recon = lg_kl = 0
        sg_recon = 0

        all_ade =[]
        all_fde =[]
        sg_ade=[]
        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, seq_start_end,
                 obs_frames, pred_frames, map_path, inv_h_t,
                 local_map, local_ic, local_homo) = batch
                batch_size = obs_traj.size(1)
                total_traj += fut_traj.size(1)

                obs_heat_map, fut_heat_map = self.make_heatmap(local_ic, local_map)
                lg_heat_map = torch.tensor(fut_heat_map[:, [0, 12]]).float().to(self.device)
                sg_heat_map = torch.tensor(fut_heat_map[:, np.concatenate([[0], self.sg_idx + 1])]).float().to(self.device)


                self.lg_cvae.forward(obs_heat_map, None, training=False)
                fut_rel_pos_dist20 = []
                pred_sg_wc20 = []
                for _ in range(20):
                    # -------- long term goal --------
                    pred_lg_heat = F.sigmoid(self.lg_cvae.sample(testing=True))

                    # -------- short term goal --------
                    # obs_lg_heat = torch.cat([obs_heat_map, pred_lg_heat[:, -1].unsqueeze(1)], dim=1)
                    pred_sg_heat = F.sigmoid(self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat[:, -1].unsqueeze(1)], dim=1), training=False))

                    # -------- trajectories --------
                    (hx, mux, log_varx) \
                        = self.encoderMx(obs_traj, seq_start_end, self.sg_unet.enc_feat, local_homo)
                    p_dist = Normal(mux, torch.sqrt(torch.exp(log_varx)))

                    pred_sg_wc = []
                    for i in range(batch_size):
                        pred_sg_ic = []
                        for heat_map in pred_sg_heat[i, 1:]:
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
                    pred_sg_wc20.append(pred_sg_wc)

                    # NO TF, pred_goals, z~prior
                    fut_rel_pos_dist_prior = self.decoderMy(
                        obs_traj[-1],
                        hx,
                        p_dist.rsample(),
                        pred_sg_wc,  # goal
                        self.sg_idx-3
                    )
                    fut_rel_pos_dist20.append(fut_rel_pos_dist_prior)



                if loss:
                    self.lg_cvae.forward(obs_heat_map, lg_heat_map, training=True)

                    lg_kl += self.lg_cvae.kl_divergence(analytic=True).sum().div(batch_size)
                    lg_recon += self.recon_loss_with_logit(input=pred_lg_heat, target=lg_heat_map).sum().div(
                        np.prod([*lg_heat_map.size()[:3]]))
                    # lg_elbo = -lg_recon_loss - self.lg_kl_weight * lg_kl

                    sg_recon += self.recon_loss_with_logit(input=pred_sg_heat, target=sg_heat_map).sum().div(
                        np.prod([*sg_heat_map.size()[:3]]))


                    (muy, log_vary) \
                        = self.encoderMy(obs_traj[-1], fut_traj[:, :, 2:4], seq_start_end, hx, train=False)
                    q_dist = Normal(muy, torch.sqrt(torch.exp(log_vary)))

                    loss_recon -= fut_rel_pos_dist_prior.log_prob(fut_traj[:, :, 2:4]).sum().div(batch_size)
                    kld = kl_divergence(q_dist, p_dist).sum().div(batch_size)
                    loss_kl += torch.clamp(kld, min=0.07)
                    # print('log_likelihood:', loglikelihood.item(), ' kl:', loss_kl.item())

                    # traj_elbo = ll_prior - self.kl_weight * kld
                    # loss = - traj_elbo - lg_elbo + sg_recon_loss


                ade, fde = [], []
                for dist in fut_rel_pos_dist20:
                    pred_fut_traj=integrate_samples(dist.rsample(), obs_traj[-1, :, :2], dt=self.dt)
                    ade.append(displacement_error(
                        pred_fut_traj, fut_traj[:,:,:2], mode='raw'
                    ))
                    fde.append(final_displacement_error(
                        pred_fut_traj[-1], fut_traj[-1,:,:2], mode='raw'
                    ))
                all_ade.append(torch.stack(ade))
                all_fde.append(torch.stack(fde))
                sg_ade.append(torch.sqrt(((torch.stack(pred_sg_wc20).permute(0, 2, 1, 3)
                                           - fut_traj[self.sg_idx,:,:2].unsqueeze(0).repeat((20,1,1,1)))**2).sum(-1)).sum(1)) # 20, 3, 4, 2
                del pred_lg_heat
                del pred_sg_heat
                del obs_heat_map
                del fut_heat_map
                del lg_heat_map
                del sg_heat_map
                del fut_rel_pos_dist20
                del pred_sg_wc20
                del pred_sg_wc

            all_ade=torch.cat(all_ade, dim=1).cpu().numpy()
            all_fde=torch.cat(all_fde, dim=1).cpu().numpy()
            sg_ade=torch.cat(sg_ade, dim=1).cpu().numpy()

            ade_min = np.min(all_ade, axis=0).mean()/self.pred_len
            fde_min = np.min(all_fde, axis=0).mean()
            ade_avg = np.mean(all_ade, axis=0).mean()/self.pred_len
            fde_avg = np.mean(all_fde, axis=0).mean()
            ade_std = np.std(all_ade, axis=0).mean()/self.pred_len
            fde_std = np.std(all_fde, axis=0).mean()

            sg_ade_min = np.min(sg_ade, axis=0).mean()/len(self.sg_idx)
            sg_ade_avg = np.mean(sg_ade, axis=0).mean()/len(self.sg_idx)
            sg_ade_std = np.std(sg_ade, axis=0).mean()/len(self.sg_idx)

        self.set_mode(train=True)
        if loss:
            return ade_min, fde_min, \
                   ade_avg, fde_avg, \
                   ade_std, fde_std, \
                   sg_ade_min, sg_ade_avg, sg_ade_std, \
                   loss_recon/b, loss_kl/b, lg_recon/b, lg_kl/b, sg_recon/b
        else:
            return ade_min, fde_min, \
                   ade_avg, fde_avg, \
                   ade_std, fde_std, \
                   sg_ade_min, sg_ade_avg, sg_ade_std





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
        self.viz.close(env=self.name + '/lines', win=self.win_id['lg_recon'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['lg_kl'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['sg_recon'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['test_lg_recon'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['test_lg_kl'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['test_sg_recon'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['ade_min'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['fde_min'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['ade_avg'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['fde_avg'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['ade_std'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['fde_std'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['sg_ade_min'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['sg_ade_avg'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['sg_ade_std'])

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
        sg_ade_min = torch.Tensor(data['sg_ade_min'])
        sg_ade_avg = torch.Tensor(data['sg_ade_avg'])
        sg_ade_std = torch.Tensor(data['sg_ade_std'])
        test_loss_recon = torch.Tensor(data['test_loss_recon'])
        test_loss_kl = torch.Tensor(data['test_loss_kl'])

        lg_recon = torch.Tensor(data['lg_recon'])
        sg_recon = torch.Tensor(data['sg_recon'])
        lg_kl = torch.Tensor(data['lg_kl'])
        test_lg_recon = torch.Tensor(data['test_lg_recon'])
        test_sg_recon = torch.Tensor(data['test_sg_recon'])
        test_lg_kl = torch.Tensor(data['test_lg_kl'])


        self.viz.line(
            X=iters, Y=lg_recon, env=self.name + '/lines',
            win=self.win_id['lg_recon'], update='append',
            opts=dict(xlabel='iter', ylabel='lg_recon',
                      title='lg_recon')
        )

        self.viz.line(
            X=iters, Y=sg_recon, env=self.name + '/lines',
            win=self.win_id['sg_recon'], update='append',
            opts=dict(xlabel='iter', ylabel='sg_recon',
                      title='sg_recon')
        )


        self.viz.line(
            X=iters, Y=lg_kl, env=self.name + '/lines',
            win=self.win_id['lg_kl'], update='append',
            opts=dict(xlabel='iter', ylabel='lg_kl',
                      title='lg_kl'),
        )


        self.viz.line(
            X=iters, Y=test_lg_recon, env=self.name + '/lines',
            win=self.win_id['test_lg_recon'], update='append',
            opts=dict(xlabel='iter', ylabel='test_lg_recon',
                      title='test_lg_recon')
        )

        self.viz.line(
            X=iters, Y=test_sg_recon, env=self.name + '/lines',
            win=self.win_id['test_sg_recon'], update='append',
            opts=dict(xlabel='iter', ylabel='test_sg_recon',
                      title='test_sg_recon')
        )


        self.viz.line(
            X=iters, Y=test_lg_kl, env=self.name + '/lines',
            win=self.win_id['test_lg_kl'], update='append',
            opts=dict(xlabel='iter', ylabel='test_lg_kl',
                      title='test_lg_kl'),
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

        self.viz.line(
            X=iters, Y=sg_ade_min, env=self.name + '/lines',
            win=self.win_id['sg_ade_min'], update='append',
            opts=dict(xlabel='iter', ylabel='sg_ade_min',
                      title='sg_ade_min'),
        )
        self.viz.line(
            X=iters, Y=sg_ade_avg, env=self.name + '/lines',
            win=self.win_id['sg_ade_avg'], update='append',
            opts=dict(xlabel='iter', ylabel='sg_ade_avg',
                      title='sg_ade_avg'),
        )
        self.viz.line(
            X=iters, Y=sg_ade_std, env=self.name + '/lines',
            win=self.win_id['sg_ade_std'], update='append',
            opts=dict(xlabel='iter', ylabel='sg_ade_std',
                      title='sg_ade_std'),
        )

    def set_mode(self, train=True):

        if train:
            self.lg_cvae.train()
            self.sg_unet.train()
            self.encoderMx.train()
            self.encoderMy.train()
            self.decoderMy.train()
        else:
            self.lg_cvae.eval()
            self.sg_unet.eval()
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
        torch.save(self.lg_cvae, lg_cvae_path)
        torch.save(self.sg_unet, sg_unet_path)
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
            self.lg_cvae = torch.load(lg_cvae_path)
            self.sg_unet = torch.load(sg_unet_path)
        else:
            self.encoderMx = torch.load(encoderMx_path, map_location='cpu')
            self.encoderMy = torch.load(encoderMy_path, map_location='cpu')
            self.decoderMy = torch.load(decoderMy_path, map_location='cpu')
            self.lg_cvae = torch.load(lg_cvae_path, map_location='cpu')
            self.sg_unet = torch.load(sg_unet_path, map_location='cpu')