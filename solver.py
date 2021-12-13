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

        self.name = '%s_enc_block_%s_fcomb_block_%s_wD_%s_lr_%s_a_%s_r_%s' % \
                    (args.dataset_name, args.no_convs_per_block, args.no_convs_fcomb, args.w_dim, args.lr_VAE,
                     args.alpha, args.gamma)


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

        self.obs_len = args.obs_len
        self.pred_len = args.pred_len
        self.num_layers = args.num_layers
        self.decoder_h_dim = args.decoder_h_dim

        if self.ckpt_load_iter == 0 or args.dataset_name =='all':  # create a new model

            # input = env + 8 past / output = env + lg
            num_filters = [32,32,64,64,64]
            self.lg_cvae = ProbabilisticUnet(input_channels=3, num_classes=1, num_filters=num_filters, latent_dim=self.w_dim,
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

        if self.ckpt_load_iter != self.max_iter:
            print("Initializing train dataset")
            _, self.train_loader = data_loader(self.args, args.dataset_dir, 'train', shuffle=True)
            print("Initializing val dataset")
            _, self.val_loader = data_loader(self.args, args.dataset_dir, 'val', shuffle=True)

            print(
                'There are {} iterations per epoch'.format(len(self.train_loader.dataset) / args.batch_size)
            )
        print('...done')



    def make_heatmap(self, local_ic, local_map):
        heatmaps = []
        for i in range(len(local_ic)):
            ohm = [local_map[i, 0]]

            heat_map_traj = np.zeros((160, 160))
            for t in range(self.obs_len):
                heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 1
                # as Y-net used variance 4 for the GT heatmap representation.
            heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
            ohm.append( heat_map_traj/heat_map_traj.sum())

            # first step
            heat_map_traj = np.zeros((160, 160))
            heat_map_traj[local_ic[i, 0, 0], local_ic[i,0, 1]] = 1
            # as Y-net used variance 4 for the GT heatmap representation.
            heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
            # plt.imshow(heat_map_traj)
            ohm.append(heat_map_traj)

            # future
            heat_map_traj = np.zeros((160, 160))
            heat_map_traj[local_ic[i, -1, 0], local_ic[i,-1, 1]] = 1
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
        return heatmaps[:,:3], heatmaps[:,3:]

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

        lg_kl_weight = 0
        print('kl_w: ', lg_kl_weight)

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

            obs_heat_map, lg_heat_map =  self.make_heatmap(local_ic, local_map)

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


            lg_kl = self.lg_cvae.kl_divergence(analytic=True)
            lg_kl = torch.clamp(lg_kl, self.fb).sum().div(batch_size)

            # lg_recon_loss = self.recon_loss_with_logit(input=recon_lg_heat, target=lg_heat_map).sum().div(np.prod([*lg_heat_map.size()[:3]]))
            lg_elbo = focal_loss


            loss = - lg_elbo

            self.optim_vae.zero_grad()
            loss.backward()
            self.optim_vae.step()



            # save model parameters
            if iteration % self.ckpt_save_iter == 0:
                self.save_checkpoint(iteration)

            # (visdom) insert current line stats
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

        lg_recon = lg_kl = 0
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

                obs_heat_map, lg_heat_map = self.make_heatmap(local_ic, local_map)

                self.lg_cvae.forward(obs_heat_map, None, training=False)
                pred_lg_wc20 = []
                for _ in range(20):
                    # -------- long term goal --------
                    pred_lg_heat = F.sigmoid(self.lg_cvae.sample(testing=True))

                    pred_lg_wc = []
                    for i in range(batch_size):
                        pred_lg_ic = []
                        for heat_map in pred_lg_heat[i]:
                            pred_lg_ic.append((heat_map == torch.max(heat_map)).nonzero()[0])
                        pred_lg_ic = torch.stack(pred_lg_ic).float()

                        # ((local_ic[0,[11,15,19]] - pred_sg_ic) ** 2).sum(1).mean()
                        back_wc = torch.matmul(
                            torch.cat([pred_lg_ic, torch.ones((len(pred_lg_ic), 1)).to(self.device)], dim=1),
                            torch.transpose(local_homo[i], 1, 0))
                        pred_lg_wc.append(back_wc[0,:2] / back_wc[0,2])
                        # ((back_wc - fut_traj[[3, 7, 11], 0, :2]) ** 2).sum(1).mean()
                    pred_lg_wc = torch.stack(pred_lg_wc)
                    pred_lg_wc20.append(pred_lg_wc)

                if loss:
                    self.lg_cvae.forward(obs_heat_map, lg_heat_map, training=True)
                    pred_lg_heat = F.normalize(pred_lg_heat.view(pred_lg_heat.shape[0], -1), p=1)
                    lg_heat_map = lg_heat_map.view(lg_heat_map.shape[0], -1)

                    lg_kl += self.lg_cvae.kl_divergence(analytic=True).sum().div(batch_size)
                    lg_recon += (self.alpha * lg_heat_map * torch.log(pred_lg_heat + self.eps) * ((1 - pred_lg_heat) ** self.gamma) \
                         + (1 - self.alpha) * (1 - lg_heat_map) * torch.log(1 - pred_lg_heat + self.eps) * (pred_lg_heat ** self.gamma)).sum().div(batch_size)

                lg_fde.append(torch.sqrt(((torch.stack(pred_lg_wc20)
                                           - fut_traj[-1,:,:2].unsqueeze(0).repeat((20,1,1)))**2).sum(-1))) # 20, 3, 4, 2

            lg_fde=torch.cat(lg_fde, dim=1).cpu().numpy() # all batches are concatenated


            lg_fde_min = np.min(lg_fde, axis=0).mean()
            lg_fde_avg = np.mean(lg_fde, axis=0).mean()
            lg_fde_std = np.std(lg_fde, axis=0).mean()

        self.set_mode(train=True)
        if loss:
            return lg_fde_min, lg_fde_avg, lg_fde_std, lg_recon/b, lg_kl/b,
        else:
            return lg_fde_min, lg_fde_avg, lg_fde_std








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
        torch.save(self.lg_cvae, lg_cvae_path)

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
            self.lg_cvae = torch.load(lg_cvae_path)

        else:
            self.lg_cvae = torch.load(lg_cvae_path, map_location='cpu')
