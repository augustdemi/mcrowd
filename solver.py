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



class Solver(object):

    ####
    def __init__(self, args):

        self.args = args

        args.num_sg = args.load_e
        self.name = '%s_lr_%s_a_%s_r_%s_num_sg_%s' % \
                    (args.dataset_name, args.lr_VAE, args.alpha, args.gamma, args.num_sg)
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
        self.sg_idx = np.array(range(12))
        self.sg_idx = np.flip(11-self.sg_idx[::(12//args.num_sg)])
        print('sg: ', self.sg_idx)
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

        lg_cvae_path = 'lgcvae_enc_block_1_fcomb_block_2_wD_10_lr_0.001_lg_klw_1.0_a_0.25_r_2.0_fb_0.7_anneal_e_10_load_e_1_run_308'
        lg_cvae_path = os.path.join('ckpts', lg_cvae_path, 'iter_42880_lg_cvae.pt')
        if self.device == 'cuda':
            self.lg_cvae = torch.load(lg_cvae_path)
        else:
            self.lg_cvae = torch.load(lg_cvae_path, map_location='cpu')
        print(">>>>>>>>> Init: ", lg_cvae_path)

        if self.ckpt_load_iter == 0 or args.dataset_name =='all':  # create a new model
            num_filters = [32, 32, 64, 64, 64]
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

        # long_dtype, float_dtype = get_dtypes(args)

        if self.ckpt_load_iter != self.max_iter:
            print("Initializing train dataset")
            _, self.train_loader = data_loader(self.args, args.dataset_dir, 'train_threshold0.5', shuffle=True)
            print("Initializing val dataset")
            _, self.val_loader = data_loader(self.args, args.dataset_dir, 'val_threshold0.5', shuffle=True)

            print(
                'There are {} iterations per epoch'.format(len(self.train_loader.dataset) / args.batch_size)
            )
        print('...done')

        self.recon_loss_with_logit = nn.BCEWithLogitsLoss(size_average = False, reduce=False, reduction=None)


    def make_heatmap(self, local_ic, local_map, aug=False):
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
        if aug:
            degree = np.random.choice([0, 90, 180, -90])
            heatmaps = transforms.Compose([
                transforms.RandomRotation(degrees=(degree, degree))
            ])(heatmaps)
        return heatmaps[:,:2], heatmaps[:,2:], heatmaps[:,-1].unsqueeze(1)

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

            (obs_traj, fut_traj, _, _, seq_start_end,
             obs_frames, pred_frames, map_path, inv_h_t,
             local_map, local_ic, local_homo) = next(iterator)
            batch_size = obs_traj.size(1) #=sum(seq_start_end[:,1] - seq_start_end[:,0])

            obs_heat_map, sg_heat_map, lg_heat_map = self.make_heatmap(local_ic, local_map, aug=True)

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

            if (iteration % (iter_per_epoch * 5) == 0):
                self.save_checkpoint(iteration)

            if iteration > 0:
                if iteration == iter_per_epoch or (self.viz_on and (iteration % (iter_per_epoch * 5) == 0)):
                    lg_fde_min, lg_fde_avg, lg_fde_std, test_lg_recon, test_lg_kl, \
                            test_sg_recon_loss, sg_ade_min, sg_ade_avg, sg_ade_std = self.evaluate_dist(self.val_loader, loss=True)
                    self.line_gather.insert(iter=iteration,
                                            lg_fde_min=lg_fde_min,
                                            lg_fde_avg=lg_fde_avg,
                                            lg_fde_std=lg_fde_std,
                                            test_total_loss=0,
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

        lg_recon = lg_kl = 0
        lg_fde=[]
        lg_fde2=[]
        sg_recon=0
        sg_ade = []
        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, _, _, seq_start_end,
                 obs_frames, pred_frames, map_path, inv_h_t,
                 local_map, local_ic, local_homo) = batch
                batch_size = obs_traj.size(1)
                total_traj += fut_traj.size(1)

                obs_heat_map, sg_heat_map, lg_heat_map = self.make_heatmap(local_ic, local_map)

                self.lg_cvae.forward(obs_heat_map, None, training=False)
                pred_lg_wcs = []
                pred_sg_wcs = []
                for _ in range(5):
                    # -------- long term goal --------
                    pred_lg_heat = F.sigmoid(self.lg_cvae.sample(testing=True))

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
                        pred_lg_wc.append(back_wc[0,:2] / back_wc[0,2])
                        # ((back_wc - fut_traj[[3, 7, 11], 0, :2]) ** 2).sum(1).mean()
                    pred_lg_wc = torch.stack(pred_lg_wc)
                    pred_lg_wcs.append(pred_lg_wc)

                    # -------- short term goal --------
                    pred_lg_heat_from_ic = []
                    for coord in pred_lg_ics:
                        heat_map_traj = np.zeros((160, 160))
                        heat_map_traj[int(coord[0,0]), int(coord[0,1])] = 1
                        heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
                        pred_lg_heat_from_ic.append(heat_map_traj)
                    pred_lg_heat_from_ic = torch.tensor(np.stack(pred_lg_heat_from_ic)).unsqueeze(1).float().to(self.device)

                    pred_sg_heat = F.sigmoid(self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat_from_ic], dim=1)))
                    # pred_sg_heat = F.sigmoid(self.sg_unet.forward(torch.cat([obs_heat_map, pred_lg_heat], dim=1)))

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

                lg_fde.append(torch.sqrt(((torch.stack(pred_lg_wcs)
                                           - fut_traj[-1,:,:2].unsqueeze(0).repeat((5,1,1)))**2).sum(-1))) # 20, 3, 4, 2
                lg_fde2.append(torch.sqrt(((torch.stack(pred_sg_wcs)[:,:,-1]
                                           - fut_traj[-1,:,:2].unsqueeze(0).repeat((5,1,1)))**2).sum(-1)))
                sg_ade.append(torch.sqrt(((torch.stack(pred_sg_wcs).permute(0, 2, 1, 3)
                                           - fut_traj[list(self.sg_idx), :, :2].unsqueeze(0).repeat(
                    (5, 1, 1, 1))) ** 2).sum(-1)).sum(1))

            lg_fde=torch.cat(lg_fde, dim=1).cpu().numpy() # all batches are concatenated
            lg_fde2=torch.cat(lg_fde2, dim=1).cpu().numpy() # all batches are concatenated
            sg_ade=torch.cat(sg_ade, dim=1).cpu().numpy()


            lg_fde_min2 = np.min(lg_fde2, axis=0).mean()
            lg_fde_avg2 = np.mean(lg_fde2, axis=0).mean()
            lg_fde_std2 = np.std(lg_fde2, axis=0).mean()

            lg_fde_min = np.min(lg_fde, axis=0).mean()
            lg_fde_avg = np.mean(lg_fde, axis=0).mean()
            lg_fde_std = np.std(lg_fde, axis=0).mean()
            sg_ade_min = np.min(sg_ade, axis=0).mean()/len(self.sg_idx)
            sg_ade_avg = np.mean(sg_ade, axis=0).mean()/len(self.sg_idx)
            sg_ade_std = np.std(sg_ade, axis=0).mean()/len(self.sg_idx)

        self.set_mode(train=True)
        if loss:
            return lg_fde_min, lg_fde_avg, lg_fde_std, lg_recon/b, lg_kl/b,\
                   sg_recon/b, sg_ade_min, sg_ade_avg, sg_ade_std
        else:
            return lg_fde_min, lg_fde_avg, lg_fde_std, \
                   lg_fde_min2, lg_fde_avg2, lg_fde_std2, \
                   sg_ade_min, sg_ade_avg, sg_ade_std




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
            self.sg_unet = torch.load(sg_unet_path)
        else:
            sg_unet_path = 'ckpts/sg_enc_block_1_fcomb_block_2_wD_10_lr_0.001_lg_klw_1_a_0.25_r_2.0_fb_2.0_anneal_e_10_load_e_1_run_1/iter_20000_sg_unet.pt'
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