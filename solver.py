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

# from model_map_ae import Decoder as Map_Decoder
from unet.probabilistic_unet import ProbabilisticUnet
from data.nuscenes.config import Config
from data.nuscenes_dataloader import data_generator

import numpy as np
import visdom
import cv2
from unet.utils import init_weights
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

        self.name = '%s_enc_block_%s_fcomb_block_%s_wD_%s_lr_%s_lg_klw_%s_a_%s_r_%s_fb_%s_anneal_e_%s_aug_%s_llprior_%s' % \
                    (args.dataset_name, args.no_convs_per_block, args.no_convs_fcomb, args.w_dim, args.lr_VAE,
                     args.lg_kl_weight, args.alpha, args.gamma, args.fb, args.anneal_epoch, args.aug, args.ll_prior_w)

        # to be appended by run_id

        # self.use_cuda = args.cuda and torch.cuda.is_available()
        self.fb = args.fb
        self.anneal_epoch = args.anneal_epoch
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.aug = args.aug
        self.device = args.device
        self.temp=1.99
        self.dt=0.5
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

        if self.ckpt_load_iter == 0 or args.dataset_name =='all':  # create a new model

            #
            # # input = env + 8 past / output = env + lg

            if args.load_e > 0:
                lg_cvae_path = 'ckpts/nu.lgcvae.ae_enc_block_1' + \
                               '_fcomb_block_2' + \
                               '_wD_20_lr_0.001_a_0.25_r_2.0_aug_1_run_2/' + \
                               'iter_14000_lg_cvae.pt'

                if self.device == 'cuda':
                    self.lg_cvae = torch.load(lg_cvae_path)
                else:
                    self.lg_cvae = torch.load(lg_cvae_path, map_location='cpu')

                print(">>>>>>>>> Init: ", lg_cvae_path)

                # random init after latent space
                for m in self.lg_cvae.unet.upsampling_path:
                    m.apply(init_weights)
                self.lg_cvae.fcomb.apply(init_weights)
                # kl weight
                self.lg_cvae.beta = args.lg_kl_weight

            else:
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
        # args.batch_size=4

        # long_dtype, float_dtype = get_dtypes(args)

        if self.ckpt_load_iter != self.max_iter:
            cfg = Config('nuscenes', False, create_dirs=True)
            torch.set_default_dtype(torch.float32)
            log = open('log.txt', 'a+')
            self.train_loader = data_generator(cfg, log, split='train', phase='training',
                                               batch_size=args.batch_size, device=self.device, scale=args.scale, shuffle=True)
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
                heat_map_traj = np.zeros_like(local_map[i])
                heat_map_traj[local_ic[i, -1, 0], local_ic[i, -1, 1]] = 1
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
                heat_map_traj = np.zeros_like(local_map[i])
                heat_map_traj[local_ic[i, -1, 0], local_ic[i, -1, 1]] = 1000
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
        return heat_maps[:,:2], heat_maps[:,2:]



    ####
    def train(self):
        self.set_mode(train=True)
        data_loader = self.train_loader

        iter_per_epoch = len(data_loader.idx_list)
        start_iter = self.ckpt_load_iter + 1
        epoch = int(start_iter / iter_per_epoch)


        lg_kl_weight = self.lg_kl_weight
        if self.anneal_epoch > 0:
            lg_kl_weight = 0
        print('>>>>>>>> kl_w: ', lg_kl_weight)

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
                if self.anneal_epoch > 0:
                    lg_kl_weight = min(self.lg_kl_weight * (epoch / self.anneal_epoch), self.lg_kl_weight)
                    print('>>>>>>>> kl_w: ', lg_kl_weight)

            # ============================================
            #          TRAIN THE VAE (ENC & DEC)
            # ============================================

            (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
             maps, local_map, local_ic, local_homo) = data
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
            recon_lg_heat = F.normalize(F.sigmoid(recon_lg_heat).view(recon_lg_heat.shape[0],-1), p=1)
            lg_heat_map= lg_heat_map.view(lg_heat_map.shape[0], -1)

            # Focal loss:
            # alpha to handle the imblanced classes: α for positive(foreground) class and 1-α for negative(background) class.
            # gamma to handle the hard positive/negative, i.e., the misclassified negative/positivle examples.
            focal_loss = (self.alpha * lg_heat_map * torch.log(recon_lg_heat + self.eps) * ((1 - recon_lg_heat) ** self.gamma) \
                         + (1 - self.alpha) * (1 - lg_heat_map) * torch.log(1 - recon_lg_heat + self.eps) * (
                recon_lg_heat ** self.gamma)).sum().div(batch_size)

            if self.ll_prior_w > 0:
                pred_lg_heat = self.lg_cvae.sample(testing=False)
                pred_lg_heat = F.normalize(F.sigmoid(pred_lg_heat).view(pred_lg_heat.shape[0], -1), p=1)
                focal_loss += self.ll_prior_w * (
                self.alpha * lg_heat_map * torch.log(pred_lg_heat + self.eps) * ((1 - pred_lg_heat) ** self.gamma) \
                + (1 - self.alpha) * (1 - lg_heat_map) * torch.log(1 - pred_lg_heat + self.eps) * (
                    pred_lg_heat ** self.gamma)).sum().div(batch_size)

            lg_kl = self.lg_cvae.kl_divergence(analytic=True)
            lg_kl = torch.clamp(lg_kl, self.fb).sum().div(batch_size)

            lg_elbo = focal_loss - lg_kl_weight * lg_kl

            loss = - lg_elbo

            self.optim_vae.zero_grad()
            loss.backward()
            self.optim_vae.step()



            # save model parameters
            if iteration % self.ckpt_save_iter == 0:
                self.save_checkpoint(iteration)

            # (visdom) insert current line stats
            # if iteration < 5000 or iteration > 14000:
            if self.viz_on and (iteration % self.viz_ll_iter == 0):
                lg_fde_min, lg_fde_avg, lg_fde_std, test_lg_recon, test_lg_kl = self.evaluate_dist(self.val_loader, loss=True)
                test_total_loss = test_lg_recon - lg_kl_weight * test_lg_kl
                self.line_gather.insert(iter=iteration,
                                        lg_fde_min=lg_fde_min,
                                        lg_fde_avg=lg_fde_avg,
                                        lg_fde_std=lg_fde_std,
                                        total_loss=-loss.item(),
                                        lg_recon=-focal_loss.item(),
                                        lg_kl=lg_kl.item(),
                                        test_total_loss=test_total_loss.item(),
                                        test_lg_recon=-test_lg_recon.item(),
                                        test_lg_kl=test_lg_kl.item(),
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

                obs_heat_map, lg_heat_map = self.make_heatmap(local_ic, local_map)

                self.lg_cvae.forward(obs_heat_map, None, training=False)
                pred_lg_wc20 = []
                for _ in range(5):
                    # -------- long term goal --------
                    pred_lg_heat = F.sigmoid(self.lg_cvae.sample(testing=True))

                    pred_lg_wc = []
                    for i in range(batch_size):
                        map_size = local_map[i].shape
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
                    lg_kl += self.lg_cvae.kl_divergence(analytic=True).sum().div(batch_size)
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
            return lg_fde_min, lg_fde_avg, lg_fde_std, lg_recon/b, lg_kl/b,
        else:
            return lg_fde_min, lg_fde_avg, lg_fde_std



    def repeat_evaluate_dist(self, data_loader):
        self.set_mode(train=False)
        for iteration in range(10000, 40000, 2000):
            lg_cvae_path = os.path.join(
                self.ckpt_dir,
                'iter_%s_lg_cvae.pt' % iteration
            )
            self.lg_cvae = torch.load(lg_cvae_path)

            lg_fde_min, lg_fde_avg, lg_fde_std= self.evaluate_dist(data_loader, loss=False)
            self.line_gather.insert(iter=iteration,
                                    lg_fde_min=lg_fde_min,
                                    lg_fde_avg=lg_fde_avg,
                                    lg_fde_std=lg_fde_std,
                                    total_loss=0,
                                    lg_recon=0,
                                    lg_kl=0,
                                    test_total_loss=0,
                                    test_lg_recon=0,
                                    test_lg_kl=0,
                                    )
            print('>>>> iter: ', iteration)
            print(lg_fde_min, lg_fde_avg, lg_fde_std)

    def check_feat(self, data_loader):
        self.set_mode(train=False)

        with torch.no_grad():
            b = 0
            for batch in data_loader:
                b += 1
                (obs_traj, fut_traj, seq_start_end,
                 obs_frames, pred_frames, map_path, inv_h_t,
                 local_map, local_ic, local_homo) = batch

                obs_heat_map, lg_heat_map = self.make_heatmap(local_ic, local_map)

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
                gara = np.zeros((160, 160))
                w = 3
                gara[:w, :] = 1
                gara[:, -w:] = 1
                gara[:, :w] = 1
                gara[-w:, :] = 1
                gara = torch.tensor(gara).float()
                obs_heat_map[0, 0] = gara

                self.lg_cvae.forward(obs_heat_map, None, training=False)

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

                #------- plot -----------
                env = local_map[i][0]
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
                                        size=local_map[i][0].shape, mode='bicubic',
                                        align_corners=False).squeeze(0).squeeze(0).detach().cpu().numpy().copy()
                    all_pred.append(1 - a / a.max())
                for k in range(5):
                    a = mmm[k][i, 0].detach().cpu().numpy().copy()
                    a = nnf.interpolate(torch.tensor(a).unsqueeze(0).unsqueeze(0),
                                        size=local_map[i][0].shape, mode='bicubic',
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
            X=iters, Y=lg_recon, env=self.name + '/lines',
            win=self.win_id['lg_recon'], update='append',
            opts=dict(xlabel='iter', ylabel='lg_recon',
                      title='lg_recon')
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
            # lg_cvae_path = 'D:\crowd\mcrowd\ckpts\lgcvae_enc_block_1_fcomb_block_2_wD_20_lr_0.001_lg_klw_1_a_0.25_r_2.0_fb_0.5_anneal_e_0_load_e_1_run_24/iter_57100_lg_cvae.pt'
            self.lg_cvae = torch.load(lg_cvae_path, map_location='cpu')

