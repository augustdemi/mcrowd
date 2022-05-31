import os
import random

import torch.optim as optim
# -----------------------------------------------------------------------------#
from utils import DataGather, mkdirs, grid2gif2, apply_poe, sample_gaussian, sample_gumbel_softmax
from model_map_ae import *
from data.ae_loader import data_loader
from scipy.ndimage import binary_dilation
from scipy import ndimage

import matplotlib.pyplot as plt
from torchvision.utils import save_image
from data.loader import data_loader
from scipy.ndimage import binary_dilation
from unet.unet import Unet
import cv2
import torch.nn.functional as F
from torchvision import transforms
from data.trajectories import seq_collate
import numpy as np

###############################################################################

def trajectory_curvature(t):
    path_distance = np.linalg.norm(t[-1] - t[0])

    lengths = np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1))  # Length between points
    path_length = np.sum(lengths)
    if np.isclose(path_distance, 0.):
        return 0, 0, 0
    return (path_length / path_distance) - 1

class Solver(object):

    ####
    def __init__(self, args):

        self.args = args

        self.name = '%s_lr_%s_a_%s_r_%s_k_%s' % \
                    (args.dataset_name, args.lr_VAE, args.alpha, args.gamma, args.k_fold)

        self.device = args.device
        self.temp=0.66
        self.dt=0.4
        self.eps=1e-9
        self.alpha=args.alpha
        self.gamma=args.gamma

        self.max_iter = int(args.max_iter)

        # do it every specified iters
        self.print_iter = args.print_iter
        self.ckpt_save_iter = args.ckpt_save_iter
        self.output_save_iter = args.output_save_iter

        # data info
        args.dataset_dir = os.path.join(args.dataset_dir, str(args.k_fold))

        self.dataset_dir = args.dataset_dir
        self.dataset_name = args.dataset_name

        # self.N = self.latent_values.shape[0]
        # self.eval_metrics_iter = args.eval_metrics_iter

        # networks and optimizers
        self.batch_size = args.batch_size
        self.lr_VAE = args.lr_VAE
        self.beta1_VAE = args.beta1_VAE
        self.beta2_VAE = args.beta2_VAE
        print(args.desc)


        # set run id
        self.run_id = args.run_id

        # finalize name
        self.name = self.name + '_run_' + str(self.run_id)

        # records (text file to store console outputs)
        self.record_file = 'records/%s.txt' % self.name

        # checkpoints
        self.ckpt_dir = os.path.join("ckpts", self.name)

        # outputs
        self.output_dir_recon = os.path.join("outputs", self.name + '_recon')

        #### create a new model or load a previously saved model

        self.ckpt_load_iter = args.ckpt_load_iter

        self.obs_len = args.obs_len
        self.pred_len = args.pred_len


        # visdom setup
        self.viz_on = args.viz_on
        if self.viz_on:
            self.win_id = dict(
                map_loss='win_map_loss', test_map_loss='win_test_map_loss'
            )
            self.line_gather = DataGather(
                'iter', 'loss', 'test_loss'
            )

            import visdom

            self.viz_port = args.viz_port  # port number, eg, 8097
            self.viz = visdom.Visdom(port=self.viz_port, env=self.name)
            self.viz_ll_iter = args.viz_ll_iter
            self.viz_la_iter = args.viz_la_iter

            self.viz_init()

        # create dirs: "records", "ckpts", "outputs" (if not exist)
        mkdirs("records");
        mkdirs("ckpts");
        mkdirs("outputs")

        if self.ckpt_load_iter == 0 or args.dataset_name =='all':  # create a new model
            # self.encoder = Encoder(
            #     fc_hidden_dim=args.hidden_dim,
            #     output_dim=args.latent_dim,
            #     drop_out=args.dropout_map).to(self.device)
            #
            # self.decoder = Decoder(
            #     fc_hidden_dim=args.hidden_dim,
            #     input_dim=args.latent_dim).to(self.device)

            num_filters = [32, 32, 32, 64, 64, 32, 32]
            # input = env + 8 past + lg / output = env + sg(including lg)
            self.sg_unet = Unet(input_channels=1, num_classes=1, num_filters=num_filters,
                             apply_last_layer=True, padding=True).to(self.device)


        else:  # load a previously saved model
            print('Loading saved models (iter: %d)...' % self.ckpt_load_iter)
            self.load_checkpoint()
            print('...done')


        # get VAE parameters
        # vae_params = \
        #     list(self.encoder.parameters()) + \
        #     list(self.decoder.parameters())
        vae_params = \
            list(self.sg_unet.parameters())

        # create optimizers
        self.optim_vae = optim.Adam(
            vae_params,
            lr=self.lr_VAE,
            betas=[self.beta1_VAE, self.beta2_VAE]
        )

        # prepare dataloader (iterable)
        print('Start loading data...')
        if self.ckpt_load_iter != self.max_iter:
            print("Initializing train dataset")
            _, self.train_loader = data_loader(self.args, args.dataset_dir, 'train', shuffle=True)
            print("Initializing val dataset")
            self.args.batch_size = 1
            _, self.val_loader = data_loader(self.args, args.dataset_dir, 'test', shuffle=False)
            self.args.batch_size = args.batch_size

            print(
                'There are {} iterations per epoch'.format(len(self.train_loader.dataset) / args.batch_size)
            )
        print('...done')


    def preprocess_map(self, local_map, aug=False):
        local_map = torch.from_numpy(local_map).float().to(self.device)

        if aug:
            all_heatmaps = []
            for h in local_map:
                h = torch.tensor(h).float().to(self.device)
                degree = np.random.choice([0, 90, 180, -90])
                all_heatmaps.append(
                    transforms.Compose([
                        transforms.RandomRotation(degrees=(degree, degree))
                    ])(h)
                )
            all_heatmaps = torch.stack(all_heatmaps)
        else:
            all_heatmaps = local_map
        return all_heatmaps



    def make_heatmap(self, local_ic, local_map, aug=False):
        heatmaps = []
        for i in range(len(local_ic)):
            ohm = [local_map[i, 0]]

            heat_map_traj = np.zeros((192, 192))
            for t in range(self.obs_len):
                heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 1
                # as Y-net used variance 4 for the GT heatmap representation.
            heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
            ohm.append(heat_map_traj / heat_map_traj.sum())

            heat_map_traj = np.zeros((192, 192))
            heat_map_traj[local_ic[i, -1, 0], local_ic[i, -1, 1]] = 1
            # as Y-net used variance 4 for the GT heatmap representation.
            heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
            # plt.imshow(heat_map_traj)
            ohm.append(heat_map_traj)

            heatmaps.append(np.stack(ohm))
            '''
            heat_map_traj = np.zeros((192, 192))
            # for t in range(self.obs_len + self.pred_len):
            for t in [0,1,2,3,4,5,6,7,11,14,17]:
                heat_map_traj[local_ic[i, t, 0], local_ic[i, t, 1]] = 1
                # as Y-net used variance 4 for the GT heatmap representation.
            heat_map_traj = ndimage.filters.gaussian_filter(heat_map_traj, sigma=2)
            plt.imshow(heat_map_traj)
            '''
        if aug:
            all_heatmaps = []
            for h in heatmaps:
                h = torch.tensor(h).float().to(self.device)
                degree = np.random.choice([0, 90, 180, -90])
                all_heatmaps.append(
                    transforms.Compose([
                        transforms.RandomRotation(degrees=(degree, degree))
                    ])(h)
                )
            all_heatmaps = torch.stack(all_heatmaps)
        else:
            all_heatmaps = torch.tensor(np.stack(heatmaps)).float().to(self.device)
        return all_heatmaps[:, :2], all_heatmaps[:, 2:]


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


            (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
             obs_frames, pred_frames, map_path, inv_h_t,
             local_map, local_ic, local_homo) = next(iterator)

            sampled_local_map = []
            for s, e in seq_start_end:
                rng = list(range(s,e))
                random.shuffle(rng)
                sampled_local_map.append(local_map[rng[:2]])

            sampled_local_map = np.concatenate(sampled_local_map)

            batch_size = sampled_local_map.shape[0]

            local_map = self.preprocess_map(sampled_local_map, aug=True)

            recon_local_map = self.sg_unet.forward(local_map)
            recon_local_map = F.sigmoid(recon_local_map)


            focal_loss =  F.mse_loss(recon_local_map, local_map).sum().div(batch_size)

            self.optim_vae.zero_grad()
            focal_loss.backward()
            self.optim_vae.step()


            # save model parameters
            if (iteration % (iter_per_epoch*10) == 0):
                self.save_checkpoint(epoch)

            # (visdom) insert current line stats
            if iteration == iter_per_epoch or (self.viz_on and (iteration % (iter_per_epoch * 10) == 0)):
                test_recon_map_loss = self.test()
                self.line_gather.insert(iter=epoch,
                                        loss=focal_loss.item(),
                                        test_loss= test_recon_map_loss.item(),
                                        )
                prn_str = ('[iter_%d (epoch_%d)] loss: %.3f \n'
                          ) % \
                          (iteration, epoch,
                           focal_loss.item())

                print(prn_str)
                self.visualize_line()
                self.line_gather.flush()


    def test(self):
        self.set_mode(train=False)
        loss=0
        b = 0
        with torch.no_grad():
            for abatch in self.val_loader:
                b += 1

                (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
                 obs_frames, pred_frames, map_path, inv_h_t,
                 local_map, local_ic, local_homo) = abatch
                batch_size = obs_traj.size(1)  # =sum(seq_start_end[:,1] - seq_start_end[:,0])
                local_map = self.preprocess_map(local_map, aug=False)

                recon_local_map = self.sg_unet.forward(local_map)
                recon_local_map = F.sigmoid(recon_local_map)

                focal_loss = F.mse_loss(recon_local_map, local_map).sum().div(batch_size)

                loss += focal_loss
        self.set_mode(train=True)
        return loss.div(b)

    ####

    def make_feat(self, test_loader):
        from sklearn.manifold import TSNE
        # from data.trajectories import seq_collate

        # from data.macro_trajectories import TrajectoryDataset
        # from torch.utils.data import DataLoader

        # test_dset = TrajectoryDataset('../datasets/large_real/Trajectories', data_split='test', device=self.device)
        # test_loader = DataLoader(dataset=test_dset, batch_size=1,
        #                              shuffle=True, num_workers=0)

        self.set_mode(train=False)
        with torch.no_grad():
            test_enc_feat = []
            total_scenario = []
            obst_ratio = []
            n_agent = []
            non_linear = []

            b = 0
            for batch in test_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
                 obs_frames, fut_frames, map_path, inv_h_t,
                 local_map, local_ic, local_homo) = batch
                n_agent.append([len(map_path)]*len(map_path))

                obs_heat_map, lg_heat_map = self.make_heatmap(local_ic, local_map, aug=True)

                # plt.imshow(local_map[0, 0])
                # plt.scatter(local_ic[0,8:,1], local_ic[0,8:,0])

                # import cv2
                # image = cv2.imread('C:\dataset\large_real\Trajectories/2.png')
                # binary = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                '''
                binary = local_map[0,0].copy().astype('uint8')
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # drawing = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
                # CountersImg = cv2.drawContours(drawing, contours, -1, (255, 255, 0), 3)
                CountersImg = cv2.drawContours(np.zeros((binary.shape[0], binary.shape[1]), dtype=np.uint8), contours, -1, (1,1), 2)
                fig = plt.figure(figsize=(10, 10))
                fig.tight_layout()
                ax = fig.add_subplot(1,2,1)
                ax.imshow(local_map[0,0])
                ax = fig.add_subplot(1, 2, 2)
                ax.imshow(CountersImg)
                print(map_path)
                '''


                for i in range(len(map_path)):
                    gt_xy = torch.cat([obs_traj[:, i, :2], fut_traj[:, i, :2]]).detach().cpu().numpy()
                    c = np.round(trajectory_curvature(gt_xy), 4)
                    non_linear.append(min(c, 10))

                for m in local_map:
                    binary = m[0].astype('uint8')
                    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    CountersImg = cv2.drawContours(np.zeros((binary.shape[0], binary.shape[1]), dtype=np.uint8),
                                                   contours, -1, (1, 1), 2)
                    obst_ratio.append(np.sum(CountersImg) / (192**2))

                for m in map_path:
                    total_scenario.append(int(m.split('/')[-1].split('.')[0])// 10)

            all_data = np.stack(
                [total_scenario, obst_ratio, non_linear, n_agent]).T
            import pandas as pd
            pd.DataFrame(all_data).to_csv('large_contour_ratio_k0_te.csv')

            print('done')
            '''
            data = pd.read_csv('C:\dataset\large_real/large_obs_ratio_k0_tr.csv')
            tr_data = np.array(pd.read_csv('C:\dataset\large_real/large_obs_ratio_k0_tr.csv'))
            te_data = np.array(pd.read_csv('C:\dataset\large_real/large_obs_ratio_k0_te.csv'))
            te= te_data[:,-1] / (192**2)
            tr= tr_data[:,-1] / (192**2)

            x = np.linspace(0,0.5,10)
            y_bot = np.linspace(30, 50, 10)
            y_dif = np.linspace(10, 5, 10)

            plt.scatter(te_data[:,1], te, s=1)
            plt.scatter(tr_data[:,1], tr, s=1)


            te_df = pd.read_csv('C:\dataset\large_real/large_obs_ratio_k0_te.csv')
            tr_df = pd.read_csv('C:\dataset\large_real/large_obs_ratio_k0_tr.csv')
            te_df['1'] = te_df['1']/(192**2)

            plt.scatter(te_df['0'], te_df['1'], s=1)

            te_df['1'] = te_df['1']*100 //10
            ax = te_df['1'].value_counts().plot(kind='bar',
                                                figsize=(14, 8),
                                                title="Number for each Owner Name")

            labels= te*100 //10


            tr_df = pd.read_csv('C:\dataset\large_real/large_obs_ratio_k0_tr.csv')
            tr_df['1'] = tr_df['1']/(192**2)
            tr_df['1'] = tr_df['1']*100 //10
            ax = tr_df['1'].value_counts().plot(kind='bar',
                                                figsize=(14, 8),
                                                title="Number for each Owner Name")

            import pandas as  pd
            df = pd.read_csv('C:\dataset\large_real/large_5_bs1.csv')
            data = np.array(df)


            # all_feat = np.load('large_tsne_ae1_tr.npy')
            all_feat_tr = np.load('large_tsne_lg_r10_k3_tr.npy')
            all_feat_te = np.load('large_tsne_lg_r10_k5_te.npy')
            # tsne_faet = np.concatenate([all_feat[:,:2], all_feat_te[:,:2]])
            all_feat = np.concatenate([all_feat_tr[:,:-1], all_feat_te[:,:-1]])
            tsne = TSNE(n_components=2, random_state=0, perplexity=30)
            tsne_feat = tsne.fit_transform(all_feat)


            # tsne_faet = all_feat_tr[:,:-3]
            # obst_ratio = all_feat_tr[:,-3]
            # curv = all_feat_tr[:,-2]
            # scenario = all_feat_tr[:,-1]

            tsne_faet = all_feat_tr[:,:-3]
            obst_ratio = all_feat_tr[:,-3]
            curv = all_feat_tr[:,-2]
            scenario =  np.concatenate([all_feat_tr[:,-1], all_feat_te[:,-1]])
            labels = scenario //10

            labels = obst_ratio*100 //10
            # labels = curv*100 //10

            target_names = ['Training', 'Test']
            colors = np.array(['blue', 'red'])
            labels= np.array(df['0.5']) // 10
            labels= np.array(df['# agent']) //10
            labels= np.array(df['curvature'])*100 //10
            labels= np.array(df['map ratio'])*100 //10


            ## k fold labels
            k=0
            labels = scenario //10
            for i in range(len(labels)):
                if labels[i] in range(k*3,(k+1)*3):
                    labels[i] = 1
                else:
                    labels[i] = 0



            # colors = ['red', 'magenta', 'lightgreen', 'slateblue', 'blue', 'darkgreen', 'darkorange',
            #           'gray', 'purple', 'turquoise', 'midnightblue', 'olive', 'black', 'pink', 'burlywood',
            #           'yellow']

            colors = np.array(['gray','pink', 'orange', 'magenta', 'darkgreen', 'cyan', 'blue', 'red', 'lightgreen', 'olive', 'burlywood', 'purple'])
            target_names = np.unique(labels)

            fig = plt.figure(figsize=(5,4))
            fig.tight_layout()

            # labels = np.concatenate([np.zeros(len(all_feat_tr)), np.ones(len(all_feat_te))])
            target_names = ['Training', 'Test']
            colors = np.array(['blue', 'red'])

            for color, i, target_name in zip(colors, np.unique(labels), target_names):
                plt.scatter(tsne_feat[labels == i, 0], tsne_feat[labels == i, 1], alpha=.5, color=color,
                            label=str(target_name), s=10)
            fig.axes[0]._get_axis_list()[0].set_visible(False)
            fig.axes[0]._get_axis_list()[1].set_visible(False)
            plt.legend(loc=0, shadow=False, scatterpoints=1)
            '''

    ####
    def viz_init(self):
        self.viz.close(env=self.name, win=self.win_id['test_map_loss'])
        self.viz.close(env=self.name, win=self.win_id['map_loss'])

    ####
    def visualize_line(self):

        # prepare data to plot
        data = self.line_gather.data
        iters = torch.Tensor(data['iter'])
        test_map_loss = torch.Tensor(data['test_loss'])
        map_loss = torch.Tensor(data['loss'])

        self.viz.line(
            X=iters, Y=map_loss, env=self.name,
            win=self.win_id['map_loss'], update='append',
            opts=dict(xlabel='iter', ylabel='loss',
                      title='Recon. map loss')
        )


        self.viz.line(
            X=iters, Y=test_map_loss, env=self.name,
            win=self.win_id['test_map_loss'], update='append',
            opts=dict(xlabel='iter', ylabel='test_loss',
                      title='Recon. map loss - Test'),
        )


    #
    #
    # def set_mode(self, train=True):
    #
    #     if train:
    #         self.encoder.train()
    #         self.decoder.train()
    #     else:
    #         self.encoder.eval()
    #         self.decoder.eval()
    #
    # ####
    # def save_checkpoint(self, iteration):
    #
    #     encoder_path = os.path.join(
    #         self.ckpt_dir,
    #         'iter_%s_encoder.pt' % iteration
    #     )
    #     decoder_path = os.path.join(
    #         self.ckpt_dir,
    #         'iter_%s_decoder.pt' % iteration
    #     )
    #
    #
    #     mkdirs(self.ckpt_dir)
    #
    #     torch.save(self.encoder, encoder_path)
    #     torch.save(self.decoder, decoder_path)
    ####


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
            sg_unet_path = 'ckpts/large.map.ae_lr_0.0001_a_0.25_r_2.0_run_8/iter_100_sg_unet.pt'
            sg_unet_path = 'ckpts/large.map.ae_lr_0.0001_a_0.25_r_2.0_k_0_run_9/iter_100_sg_unet.pt'
            print('>>>>>>>>>>> load: ', sg_unet_path)

            self.sg_unet = torch.load(sg_unet_path)
        else:
            sg_unet_path = 'ckpts/large.map.ae_lr_0.0001_a_0.25_r_2.0_run_8/iter_100_sg_unet.pt'
            sg_unet_path = 'ckpts/large.map.ae_lr_0.0001_a_0.25_r_2.0_k_1_run_10/iter_200_sg_unet.pt'
            # sg_unet_path = 'ckpts/large.map.ae_lr_0.0001_a_0.25_r_2.0_k_0_run_10/iter_200_sg_unet.pt'
            sg_unet_path = 'ckpts/large.map.ae_lr_0.0001_a_0.25_r_2.0_k_0_run_9/iter_40_sg_unet.pt'
            # sg_unet_path = 'd:\crowd\mcrowd\ckpts\mapae.path_lr_0.001_a_0.25_r_2.0_run_2/iter_3360_sg_unet.pt'
            self.sg_unet = torch.load(sg_unet_path, map_location='cpu')
         ####

    #
    # def load_checkpoint(self):
    #
    #     encoder_path = os.path.join(
    #         self.ckpt_dir,
    #         'iter_%s_encoder.pt' % self.ckpt_load_iter
    #     )
    #     decoder_path = os.path.join(
    #         self.ckpt_dir,
    #         'iter_%s_decoder.pt' % self.ckpt_load_iter
    #     )
    #
    #     if self.device == 'cuda':
    #         self.encoder = torch.load(encoder_path)
    #         self.decoder = torch.load(decoder_path)
    #     else:
    #         self.encoder = torch.load(encoder_path, map_location='cpu')
    #         self.decoder = torch.load(decoder_path, map_location='cpu')
    #
    # def load_map_weights(self, map_path):
    #     if self.device == 'cuda':
    #         loaded_map_w = torch.load(map_path)
    #     else:
    #         loaded_map_w = torch.load(map_path, map_location='cpu')
    #     self.encoder.conv1.weight = loaded_map_w.map_net.conv1.weight
    #     self.encoder.conv2.weight = loaded_map_w.map_net.conv2.weight
    #     self.encoder.conv3.weight = loaded_map_w.map_net.conv3.weight