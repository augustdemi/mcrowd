import os

import torch.optim as optim
# -----------------------------------------------------------------------------#
from utils import DataGather, mkdirs, grid2gif2, apply_poe, sample_gaussian, sample_gumbel_softmax
from model_map_ae import *
from data.ae_loader import data_loader
from scipy.ndimage import binary_dilation

import matplotlib.pyplot as plt
from torchvision.utils import save_image
from data.loader import data_loader
from scipy.ndimage import binary_dilation
from unet.unet import Unet
import cv2
import torch.nn.functional as F
from torchvision import transforms
from data.nuscenes.config import Config
from data.nuscenes_dataloader import data_generator

###############################################################################

class Solver(object):

    ####
    def __init__(self, args):

        self.args = args

        self.name = '%s_lr_%s_a_%s_r_%s' % \
                    (args.dataset_name, args.lr_VAE, args.alpha, args.gamma)

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

        #### create a new model or load a previously saved model

        self.ckpt_load_iter = args.ckpt_load_iter

        self.obs_len = args.obs_len
        self.pred_len = args.pred_len

        if self.ckpt_load_iter == 0 or args.dataset_name =='all':  # create a new model
            # self.encoder = Encoder(
            #     fc_hidden_dim=args.hidden_dim,
            #     output_dim=args.latent_dim,
            #     drop_out=args.dropout_map).to(self.device)
            #
            # self.decoder = Decoder(
            #     fc_hidden_dim=args.hidden_dim,
            #     input_dim=args.latent_dim).to(self.device)

            num_filters = [32, 32, 64, 64, 64]
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
            cfg = Config('nuscenes_train', False, create_dirs=True)
            torch.set_default_dtype(torch.float32)
            log = open('log.txt', 'a+')
            self.train_loader = data_generator(cfg, log, split='train', phase='training',
                                               batch_size=args.batch_size, device=self.device, scale=args.scale, shuffle=True)

            cfg = Config('nuscenes', False, create_dirs=True)
            torch.set_default_dtype(torch.float32)
            log = open('log.txt', 'a+')
            self.val_loader = data_generator(cfg, log, split='test', phase='testing',
                                             batch_size=args.batch_size, device=self.device, scale=args.scale, shuffle=False)

            # self.train_loader = self.val_loader

            print(
                'There are {} iterations per epoch'.format(len(self.train_loader.idx_list))
            )
        print('...done')


    def preprocess_map(self, local_map, aug=False):
        env=[]
        down_size=256
        for i in range(len(local_map)):
            map_size = local_map[i][0].shape[0]
            if map_size < down_size:
                env = np.full((down_size,down_size),1)
            else:
                env.append(cv2.resize(local_map[i][0], dsize=(down_size, down_size)))

        env = torch.tensor(env).float().to(self.device).unsqueeze(1)

        if aug:
            degree = np.random.choice([0,90,180, -90])
            env = transforms.Compose([
                transforms.RandomRotation(degrees=(degree, degree))
            ])(env)
        return env



    ####
    def train(self):
        self.set_mode(train=True)
        data_loader = self.train_loader

        iter_per_epoch = len(data_loader.idx_list)
        start_iter = self.ckpt_load_iter + 1
        epoch = int(start_iter / iter_per_epoch)


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

            # ============================================
            #          TRAIN THE VAE (ENC & DEC)
            # ============================================
            (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
             maps, local_map, local_ic, local_homo) = data

            batch_size = obs_traj.size(1) #=sum(seq_start_end[:,1] - seq_start_end[:,0])
            local_map = self.preprocess_map(local_map, aug=True)
            recon_local_map = self.sg_unet.forward(local_map)
            recon_local_map = F.sigmoid(recon_local_map)

            focal_loss = F.mse_loss(recon_local_map, local_map).sum().div(batch_size)

            self.optim_vae.zero_grad()
            focal_loss.backward()
            self.optim_vae.step()


            # save model parameters
            if (iteration % (iter_per_epoch*5) == 0):
                self.save_checkpoint(iteration)

            # (visdom) insert current line stats
            if iteration == iter_per_epoch or (self.viz_on and (iteration % (iter_per_epoch * 5) == 0)):
                test_recon_map_loss = self.test()
                self.line_gather.insert(iter=iteration,
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
            while not self.val_loader.is_epoch_end():
                data = self.val_loader.next_sample()
                if data is None:
                    continue
                b+=1
                (obs_traj, fut_traj, obs_traj_st, fut_vel_st, seq_start_end,
                 maps, local_map, local_ic, local_homo) = data
                batch_size = obs_traj.size(1)
                local_map = self.preprocess_map(local_map, aug=False)

                recon_local_map = self.sg_unet.forward(local_map)
                recon_local_map = F.sigmoid(recon_local_map)

                focal_loss = F.mse_loss(recon_local_map, local_map).sum().div(batch_size)


                loss += focal_loss
        self.set_mode(train=True)
        return loss.div(b)

    ####

    ####
    def recon(self, test_loader, train_loader):
        from sklearn.manifold import TSNE

        self.set_mode(train=False)
        with torch.no_grad():

            test_range= list(range(len(test_loader.idx_list)))
            np.random.shuffle(test_range)

            # train_range= range(len(train_loader.dataset))
            # np.random.shuffle(train_range)
            test_enc_feat = []
            train_enc_feat = []
            s = 500

            n_sample = 0
            for i in test_range:
                test_loader.index = i
                data = test_loader.next_sample()
                if data is None:
                    continue
                local_map = self.preprocess_map(data[-3][:1], aug=False)
                n_sample += len(local_map)
                self.sg_unet.forward(local_map)
                # recon_local_map = F.sigmoid(recon_local_map)
                # plt.imshow(recon_local_map[0, 0])
                test_enc_feat.append(self.sg_unet.enc_feat.view(len(local_map), -1).detach().cpu().numpy())
                if n_sample >=s:
                    break

            n_sample = 0
            for i in test_range:
                train_loader.index = i
                data = train_loader.next_sample()
                if data is None:
                    continue
                local_map = self.preprocess_map(data[-3][:1], aug=False)
                n_sample += len(local_map)
                self.sg_unet.forward(local_map)
                # recon_local_map = F.sigmoid(recon_local_map)
                # plt.imshow(recon_local_map[0, 0])
                train_enc_feat.append(self.sg_unet.enc_feat.view(len(local_map), -1).detach().cpu().numpy())
                if n_sample >=s:
                    break

            test_enc_feat = np.concatenate(test_enc_feat)[:s]
            train_enc_feat = np.concatenate(train_enc_feat)[:s]

            tsne = TSNE(n_components=2, random_state=0)
            X_r2 = tsne.fit_transform(np.concatenate([train_enc_feat, test_enc_feat]))

            X_r2 = np.load('nu_tsne2.npy')

            labels = np.concatenate([np.zeros(s), np.ones(s)])
            # target_names = np.unique(labels)
            target_names = ['Training', 'Test']
            # colors = np.array(
            #     ['burlywood', 'turquoise', 'darkorange', 'blue', 'green', 'yellow', 'red', 'black', 'purple',
            #      'magenta'])
            colors = np.array(['blue', 'red'])

            fig = plt.figure(figsize=(5,4))
            fig.tight_layout()

            for color, i, target_name in zip(colors, np.unique(labels), target_names):
                plt.scatter(X_r2[labels == i, 0], X_r2[labels == i, 1], alpha=.5, color=color,
                            label=target_name, s=5)
            fig.axes[0]._get_axis_list()[0].set_visible(False)
            fig.axes[0]._get_axis_list()[1].set_visible(False)
            plt.legend(loc=4, shadow=False, scatterpoints=1)


            ############################
            out_dir = os.path.join('./output',self.name, dset, str(self.max_iter))
            mkdirs(out_dir)
            for i in range(map.shape[0]):
                save_image(recon_map[i], str(os.path.join(out_dir, 'recon_img'+str(i)+'.png')), nrow=self.pred_len, pad_value=1)
                save_image(map[i], str(os.path.join(out_dir, 'gt_img'+str(i)+'.png')), nrow=self.pred_len, pad_value=1)

        self.set_mode(train=True)

    def local_map_navi_ratio(self, test_loader):
        self.set_mode(train=False)
        loss=0
        b = 0
        ratio = []
        with torch.no_grad():
            while not test_loader.is_epoch_end():
                data = test_loader.next_sample()
                if data is None:
                    continue
                b+=1
                local_map = data[-3]
                ratio.extend([len(np.where(m ==0)[0]) / m.shape[0]**2 for m in local_map])
        print(np.array(ratio).mean())


    ####
    def viz_init(self):
        self.viz.close(env=self.name + '/lines', win=self.win_id['test_map_loss'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['map_loss'])

    ####
    def visualize_line(self):

        # prepare data to plot
        data = self.line_gather.data
        iters = torch.Tensor(data['iter'])
        test_map_loss = torch.Tensor(data['test_loss'])
        map_loss = torch.Tensor(data['loss'])

        self.viz.line(
            X=iters, Y=map_loss, env=self.name + '/lines',
            win=self.win_id['map_loss'], update='append',
            opts=dict(xlabel='iter', ylabel='loss',
                      title='Recon. map loss')
        )

        self.viz.line(
            X=iters, Y=test_map_loss, env=self.name + '/lines',
            win=self.win_id['test_map_loss'], update='append',
            opts=dict(xlabel='iter', ylabel='test_loss',
                      title='Recon. map loss - Test'),
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
            sg_unet_path = 'd:\crowd\mcrowd\ckpts\mapae.nu_lr_0.001_a_0.25_r_2.0_run_2/iter_44940_sg_unet.pt'
            self.sg_unet = torch.load(sg_unet_path, map_location='cpu')
         ####
