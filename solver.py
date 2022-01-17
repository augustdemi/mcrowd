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
# from skimage.transform import resize
import cv2
from torch.distributions import kl_divergence
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
# from model_map_ae import Decoder as Map_Decoder
from unet.probabilistic_unet import ProbabilisticUnet
from data.nuscenes.config import Config
from data.nuscenes_dataloader import data_generator
import numpy as np
import visdom
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

        self.name = '%s_enc_block_%s_fcomb_block_%s_wD_%s_lr_%s_a_%s_r_%s_aug_%s' % \
                    (args.dataset_name, args.no_convs_per_block, args.no_convs_fcomb, args.w_dim, args.lr_VAE,
                     args.alpha, args.gamma, args.aug)


        # to be appended by run_id

        # self.use_cuda = args.cuda and torch.cuda.is_available()
        self.fb = args.fb
        self.anneal_epoch = args.anneal_epoch
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.aug = args.aug
        self.device = args.device
        self.temp=1.99
        self.dt=0.4
        self.eps=1e-9
        self.ll_prior_w =args.ll_prior_w
        self.sg_idx =  np.array([3,7,11])
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
                recon='win_recon', total_loss='win_total_loss', test_total_loss='win_test_total_loss',
                lg_recon='win_lg_recon', lg_kl='win_lg_kl',
                test_lg_recon='win_test_lg_recon', test_lg_kl='win_test_lg_kl',
                lg_fde_min='win_lg_fde_min', lg_fde_avg='win_lg_fde_avg', lg_fde_std='win_lg_fde_std'
            )
            self.line_gather = DataGather(
                'iter', 'total_loss',
                'test_total_loss',
                'lg_recon', 'lg_kl',
                'test_lg_recon', 'test_lg_kl',
                'lg_fde_min', 'lg_fde_avg', 'lg_fde_std'
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

        self.obs_len = 8
        self.pred_len = 12
        self.num_layers = args.num_layers
        self.decoder_h_dim = args.decoder_h_dim

        if self.ckpt_load_iter == 0 or args.dataset_name =='all':  # create a new model

            # input = env + 8 past / output = env + lg
            num_filters = [32,32,64,64,64]
            self.lg_cvae = ProbabilisticUnet(input_channels=2, num_classes=1, num_filters=num_filters, latent_dim=self.w_dim,
                                    no_convs_fcomb=self.no_convs_fcomb, no_convs_per_block=self.no_convs_per_block, beta=self.lg_kl_weight).to(self.device)


        else:  # load a previously saved model
            print('Loading saved models (iter: %d)...' % self.ckpt_load_iter)
            self.load_checkpoint()
            print('...done')


        # get VAE parameters
        vae_params = \
            list(self.lg_cvae.parameters())

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
            plt.imshow(local_map[i][0])
            plt.scatter(local_ic[i,:8,1], local_ic[i,:8,0], s=1, c='b')
            plt.scatter(local_ic[i,8:,1], local_ic[i,8:,0], s=1, c='r')
            '''
            map_size = local_map[i][0].shape[0]
            env = cv2.resize(local_map[i][0], dsize=(down_size, down_size))
            ohm = [env]
            heat_map_traj = np.zeros_like(local_map[i][0])
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
                                       size=local_map[i][0].shape, mode='nearest').squeeze(0).squeeze(0)
            heat_map = nnf.interpolate(torch.tensor(heat_map_traj).unsqueeze(0).unsqueeze(0),
                                       size=local_map[i][0].shape,  mode='bicubic',
                                              align_corners = False).squeeze(0).squeeze(0)
            '''
            heat_map_traj = np.zeros_like(local_map[i][0])
            heat_map_traj[local_ic[i, -1, 0], local_ic[i, -1, 1]] = 1000
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
        return heat_maps[:,:2], heat_maps[:,2:]


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
            obs_heat_map, lg_heat_map =  self.make_heatmap(local_ic, local_map, aug=self.aug)

            #-------- long term goal --------
            recon_lg_heat = self.lg_cvae.forward(obs_heat_map, lg_heat_map, training=True)
            # recon_lg_heat = F.sigmoid(recon_lg_heat)
            recon_lg_heat = F.normalize(F.sigmoid(recon_lg_heat).view(recon_lg_heat.shape[0],-1), p=1)
            lg_heat_map= lg_heat_map.view(lg_heat_map.shape[0], -1)


            # Focal loss:
            # alpha to handle the imblanced classes: α for positive(foreground) class and 1-α for negative(background) class.
            # gamma to handle the hard positive/negative, i.e., the misclassified negative/positivle examples.
            focal_loss = (self.alpha * lg_heat_map * torch.log(recon_lg_heat + self.eps) * ((1 - recon_lg_heat) ** self.gamma) \
                         + (1 - self.alpha) * (1 - lg_heat_map) * torch.log(1 - recon_lg_heat + self.eps) * (
                recon_lg_heat ** self.gamma)).sum().div(batch_size)

            lg_elbo = focal_loss


            loss = - lg_elbo

            self.optim_vae.zero_grad()
            loss.backward()
            self.optim_vae.step()



            # save model parameters
            if iteration % self.ckpt_save_iter == 0:
                self.save_checkpoint(iteration)

            # (visdom) insert current line stats
            if iteration > 0:
                if self.viz_on and (iteration % self.viz_ll_iter == 0):
                    lg_fde_min, lg_fde_avg, lg_fde_std, test_lg_recon = self.evaluate_dist(self.val_loader, loss=True)
                    test_total_loss = test_lg_recon
                    self.line_gather.insert(iter=iteration,
                                            lg_fde_min=lg_fde_min,
                                            lg_fde_avg=lg_fde_avg,
                                            lg_fde_std=lg_fde_std,
                                            total_loss=-loss.item(),
                                            lg_recon=-focal_loss.item(),
                                            lg_kl=0,
                                            test_total_loss=test_total_loss.item(),
                                            test_lg_recon=-test_lg_recon.item(),
                                            test_lg_kl=0,
                                            )

                    prn_str = ('[iter_%d (epoch_%d)] VAE Loss: %.3f '
                              ) % \
                              (iteration, epoch,
                               loss.item(),
                               )

                    print(prn_str)

                # (visdom) visualize line stats (then flush out)
                if self.viz_on and (iteration % self.viz_la_iter == 0):
                    self.visualize_line()
                    self.line_gather.flush()





    def evaluate_dist(self, data_loader, loss=False):
        self.set_mode(train=False)
        total_traj = 0

        lg_recon = 0
        lg_fde=[]
        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_st, fut_traj_st, seq_start_end,
                 obs_frames, fut_frames, map_path, inv_h_t,
                 local_map, local_ic, local_homo) = batch
                batch_size = obs_traj.size(1)
                total_traj += fut_traj.size(1)

                obs_heat_map, lg_heat_map = self.make_heatmap(local_ic, local_map)

                '''
                idx= 0
                plt.imshow(cv2.imread('C:\dataset\HTP-benchmark\A2A Data/' + map_path[idx].replace('txt', 'png')))
                all_traj = torch.cat([obs_traj[:,idx], fut_traj[:,idx]],0).numpy()[:,:2]
                all_traj = np.concatenate([all_traj, np.ones((20,1))],1)
                map_traj = np.matmul(all_traj, inv_h_t[idx])
                map_traj /= np.expand_dims(map_traj[:, 2], 1)
                plt.scatter(map_traj[:8, 0], map_traj[:8, 1], s=1, c='b')
                plt.scatter(map_traj[8:, 0], map_traj[8:, 1], s=1, c='r')
                '''

                self.lg_cvae.forward(obs_heat_map, None, training=False)
                pred_lg_wc20 = []
                for _ in range(5):
                    # -------- long term goal --------
                    pred_lg_heat = F.sigmoid(self.lg_cvae.sample(testing=True))

                    pred_lg_wc = []
                    for i in range(batch_size):
                        map_size = local_map[i][0].shape
                        h = local_homo[i]
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

                        back_wc = torch.matmul(
                            torch.cat([pred_lg_ic, torch.ones((len(pred_lg_ic), 1)).to(self.device)], dim=1),
                            torch.transpose(h, 1, 0))
                        pred_lg_wc.append(back_wc[0,:2] / back_wc[0,2])

                    pred_lg_wc = torch.stack(pred_lg_wc).squeeze(1)
                    pred_lg_wc20.append(pred_lg_wc)

                if loss:
                    self.lg_cvae.forward(obs_heat_map, lg_heat_map, training=True)
                    pred_lg_heat = F.normalize(pred_lg_heat.view(pred_lg_heat.shape[0], -1), p=1)
                    lg_heat_map = lg_heat_map.view(lg_heat_map.shape[0], -1)

                    lg_recon += (self.alpha * lg_heat_map * torch.log(pred_lg_heat + self.eps) * ((1 - pred_lg_heat) ** self.gamma) \
                         + (1 - self.alpha) * (1 - lg_heat_map) * torch.log(1 - pred_lg_heat + self.eps) * (pred_lg_heat ** self.gamma)).sum().div(batch_size)

                lg_fde.append(torch.sqrt(((torch.stack(pred_lg_wc20)
                                           - fut_traj[-1,:,:2].unsqueeze(0).repeat((5,1,1)))**2).sum(-1))) # 20, 3, 4, 2

            lg_fde=torch.cat(lg_fde, dim=1).cpu().numpy() # all batches are concatenated

            lg_fde_min = np.min(lg_fde, axis=0).mean()
            lg_fde_avg = np.mean(lg_fde, axis=0).mean()
            lg_fde_std = np.std(lg_fde, axis=0).mean()

        self.set_mode(train=True)
        if loss:
            return lg_fde_min, lg_fde_avg, lg_fde_std, lg_recon/b
        else:
            return lg_fde_min, lg_fde_avg, lg_fde_std




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



    def collision_stat(self, data_loader, threshold=0.2):
        self.set_mode(train=False)

        total_coll5 = 0
        total_coll10 = 0
        total_coll15 = 0
        total_coll20 = 0
        total_coll25 = 0
        n_scene = 0
        total_ped = []
        e_ped = []
        avg_dist = 0
        n_agent = 0
        min_dist =100000

        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_st, fut_traj_st, seq_start_end,
                 obs_frames, fut_frames, map_path, inv_h_t,
                 local_map, local_ic, local_homo) = batch
                b+=1
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
                        min_dist = min(diff_agent_dist.min(), min_dist)
                        n_agent += len(diff_agent_dist)
                        total_coll5 += (diff_agent_dist < 0.5).sum()
                        total_coll10 += (diff_agent_dist < 0.6).sum()
                        total_coll15 += (diff_agent_dist < 0.7).sum()
                        total_coll20 += (diff_agent_dist < 0.8).sum()
                        total_coll25 += (diff_agent_dist < 0.9).sum()
        print('total_coll5: ', total_coll5)
        print('total_coll10: ', total_coll10)
        print('total_coll15: ', total_coll15)
        print('total_coll20: ', total_coll20)
        print('total_coll25: ', total_coll25)
        print('n_scene: ', n_scene)
        print('scenes with ped > 1:', len(e_ped))
        print('avg_dist:', avg_dist/n_agent)
        print('min_dist:', min_dist)
        print('avg e_ped:', np.array(e_ped).mean())
        print('avg ped:', np.array(total_ped).mean())


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
                plt.imshow(local_map[i][0])

                # ----------- 12 traj
                # heat_map_traj = np.zeros((160, 160))
                heat_map_traj = local_map[i][0].detach().cpu().numpy().copy()
                # for t in range(self.obs_len):
                for t in [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 19]:
                    heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 100
                    # as Y-net used variance 4 for the GT heatmap representation.
                heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
                plt.imshow(heat_map_traj)
                # plt.imshow(np.stack([heat_map_traj, local_map[i][0]],axis=2))


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
                z[:, :32] = 8
                z[:, 32:] = -8
                pred_lg_mm = F.sigmoid(self.lg_cvae.fcomb.forward(self.lg_cvae.unet_features, z))
                # ---------- posterior
                posterior_latent_space = self.lg_cvae.posterior.forward(obs_heat_map, lg_heat_map)
                z_post = posterior_latent_space.rsample()
                pred_lg_post = F.sigmoid(self.lg_cvae.fcomb.forward(self.lg_cvae.unet_features, z_post))
                # ---------- without latetn, only feature map
                pred_lg_patch = self.lg_cvae.fcomb.last_layer(self.lg_cvae.unet_features)

                ###### =============== plot LG ==================#######
                fig = plt.figure(figsize=(8, 8))
                k = 0
                title = ['prior', 'post', '0', 'patch']

                env = local_map[i][0].detach().cpu().numpy()
                heat_map_traj = np.zeros_like(env)
                for t in [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 19]:
                    heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 20
                heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)

                for m in [pred_lg_prior, pred_lg_post, pred_lg_zeros, pred_lg_patch]:
                    ax = fig.add_subplot(2, 2, k + 1)
                    ax.set_title(title[k])
                    ax.imshow(m[i, 0])
                    # ax.imshow(np.stack([m[i, 0] / m[i, 0].max(), env, heat_map_traj],axis=2))
                    k += 1


#4444444444444444444t
                z = torch.randint(-8, 8, (30, 32))
                m1 = F.sigmoid(self.lg_cvae.fcomb.forward(self.lg_cvae.unet_features, z))

                z = torch.randint(-8, 5, (30, 32))
                m2 = F.sigmoid(self.lg_cvae.fcomb.forward(self.lg_cvae.unet_features, z))


                z = torch.randint(-5,5,(30,32))
                m3 = F.sigmoid(self.lg_cvae.fcomb.forward(self.lg_cvae.unet_features, z))


                z = torch.ones_like(z_prior)
                z[:, 32:] = -5
                z[:, :32] = 5
                m4 = F.sigmoid(self.lg_cvae.fcomb.forward(self.lg_cvae.unet_features, z))


                fig = plt.figure(figsize=(8, 8))
                title = ['prior', 'post', '0', 'patch']

                env = local_map[i][0].detach().cpu().numpy()
                heat_map_traj = np.zeros_like(env)
                for t in [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 19]:
                    heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 20
                heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)

                k = 0
                for m in [m1, m2, m3, m4]:
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
                env = local_map[i][0].detach().cpu().numpy()

                heat_map_traj = np.zeros_like(env)
                for t in [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 19]:
                    heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 20
                heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)

                for m in pred_sg_heat[i]:
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




    ####
    def viz_init(self):
        self.viz.close(env=self.name + '/lines', win=self.win_id['total_loss'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['test_total_loss'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['lg_recon'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['lg_kl'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['test_lg_recon'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['test_lg_kl'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['lg_fde_min'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['lg_fde_avg'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['lg_fde_std'])
    ####
    def visualize_line(self):

        # prepare data to plot
        data = self.line_gather.data
        iters = torch.Tensor(data['iter'])
        total_loss = torch.Tensor(data['total_loss'])
        test_total_loss = torch.Tensor(data['test_total_loss'])

        lg_fde_min = torch.Tensor(data['lg_fde_min'])
        lg_fde_avg = torch.Tensor(data['lg_fde_avg'])
        lg_fde_std = torch.Tensor(data['lg_fde_std'])

        lg_recon = torch.Tensor(data['lg_recon'])
        lg_kl = torch.Tensor(data['lg_kl'])
        test_lg_recon = torch.Tensor(data['test_lg_recon'])
        test_lg_kl = torch.Tensor(data['test_lg_kl'])


        self.viz.line(
            X=iters, Y=total_loss, env=self.name + '/lines',
            win=self.win_id['total_loss'], update='append',
            opts=dict(xlabel='iter', ylabel='elbo',
                      title='elbo')
        )

        self.viz.line(
            X=iters, Y=test_total_loss, env=self.name + '/lines',
            win=self.win_id['test_total_loss'], update='append',
            opts=dict(xlabel='iter', ylabel='elbo',
                      title='test_elbo')
        )


        self.viz.line(
            X=iters, Y=lg_kl, env=self.name + '/lines',
            win=self.win_id['lg_kl'], update='append',
            opts=dict(xlabel='iter', ylabel='lg_kl',
                      title='lg_kl'),
        )

        self.viz.line(
            X=iters, Y=lg_recon, env=self.name + '/lines',
            win=self.win_id['lg_recon'], update='append',
            opts=dict(xlabel='iter', ylabel='lg_recon',
                      title='lg_recon')
        )


        self.viz.line(
            X=iters, Y=test_lg_recon, env=self.name + '/lines',
            win=self.win_id['test_lg_recon'], update='append',
            opts=dict(xlabel='iter', ylabel='test_lg_recon',
                      title='test_lg_recon')
        )



        self.viz.line(
            X=iters, Y=test_lg_kl, env=self.name + '/lines',
            win=self.win_id['test_lg_kl'], update='append',
            opts=dict(xlabel='iter', ylabel='test_lg_kl',
                      title='test_lg_kl'),
        )



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


    def set_mode(self, train=True):
        if train:
            self.lg_cvae.train()
        else:
            self.lg_cvae.eval()

    ####
    def save_checkpoint(self, iteration):
        lg_cvae_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_lg_cvae.pt' % iteration
        )
        mkdirs(self.ckpt_dir)
        del self.lg_cvae.unet.blocks
        torch.save(self.lg_cvae, lg_cvae_path)

    ####
    def load_checkpoint(self):

        lg_cvae_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_lg_cvae.pt' % self.ckpt_load_iter
        )


        if self.device == 'cuda':
            self.lg_cvae = torch.load(lg_cvae_path)

        else:
            self.lg_cvae = torch.load(lg_cvae_path, map_location='cpu')
