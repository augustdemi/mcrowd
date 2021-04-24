import os

import torch.optim as optim
# -----------------------------------------------------------------------------#
from utils import DataGather, mkdirs, grid2gif2, apply_poe, sample_gaussian, sample_gumbel_softmax
from model_map_ae import *
from data.ae_loader import data_loader
from eval_util import ploot

import matplotlib.pyplot as plt
from torch.distributions import RelaxedOneHotCategorical as concrete
from torchvision.utils import save_image
from data.ae_map import seq_collate

###############################################################################

class Solver(object):

    ####
    def __init__(self, args):

        self.args = args

        # self.name = '%s_pred_len_%s_zS_%s_embedding_dim_%s_enc_h_dim_%s_dec_h_dim_%s_mlp_dim_%s_pool_dim_%s_lr_%s_klw_%s' % \
        #             (args.dataset_name, args.pred_len, args.zS_dim, 16, args.encoder_h_dim, args.decoder_h_dim, args.mlp_dim, args.pool_dim, args.lr_VAE, args.kl_weight)

        # self.name = '%s_pred_len_%s_zS_%s_dr_mlp_%s_dr_rnn_%s_enc_h_dim_%s_dec_h_dim_%s_mlp_dim_%s_pool_dim_%s_lr_%s_klw_%s' % \
        #             (args.dataset_name, args.pred_len, args.zS_dim, args.dropout_mlp, args.dropout_rnn, args.encoder_h_dim,
        #              args.decoder_h_dim, args.mlp_dim, 0, args.lr_VAE, args.kl_weight)

        if args.gamma==0:
            self.name = '%s_map_size_%s_drop_out%s' % \
                        (args.dataset_name, args.map_size, args.dropout_map)
        else:
            if args.alpha==0.5:
                self.name = '%s_map_size_%s_drop_out%s_gamma%s' % \
                            (args.dataset_name, args.map_size, args.dropout_map, args.gamma)
            else:
                self.name = '%s_map_size_%s_drop_out%s_gamma%s_alpha%s' % \
                            (args.dataset_name, args.map_size, args.dropout_map, args.gamma, args.alpha)

        # to be appended by run_id

        # self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = args.device
        self.temp=0.66
        self.dt=0.4
        self.eps=1e-9
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.kl_weight=args.kl_weight

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
        self.lr_VAE = args.lr_VAE
        self.beta1_VAE = args.beta1_VAE
        self.beta2_VAE = args.beta2_VAE
        print(args.desc)

        # visdom setup
        self.viz_on = args.viz_on
        if self.viz_on:
            self.win_id = dict(
                loss='win_loss', test_loss='win_test_loss',
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
            self.encoder = Encoder(
                fc_hidden_dim=32,
                output_dim=8,
                drop_out=args.dropout_map).to(self.device)

            self.decoder = Decoder(
                fc_hidden_dim=32,
                input_dim=8).to(self.device)

        else:  # load a previously saved model
            print('Loading saved models (iter: %d)...' % self.ckpt_load_iter)
            self.load_checkpoint()
            print('...done')


        # get VAE parameters
        vae_params = \
            list(self.encoder.parameters()) + \
            list(self.decoder.parameters())

        # create optimizers
        self.optim_vae = optim.Adam(
            vae_params,
            lr=self.lr_VAE,
            betas=[self.beta1_VAE, self.beta2_VAE]
        )

        # prepare dataloader (iterable)
        if args.ckpt_load_iter == args.max_iter:
            val_path = os.path.join(self.dataset_dir, self.dataset_name, 'val')
            _, self.val_loader = data_loader(self.args, val_path)

        else:
            print('Start loading data...')
            train_path = os.path.join(self.dataset_dir, self.dataset_name, 'train2')
            val_path = os.path.join(self.dataset_dir, self.dataset_name, 'val')

            # long_dtype, float_dtype = get_dtypes(args)

            print("Initializing train dataset")
            _, self.train_loader = data_loader(self.args, train_path)
            print("Initializing val dataset")
            # self.args.batch_size = 32
            _, self.val_loader = data_loader(self.args, val_path)
            # self.val_loader = self.train_loader

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
            (obs_traj, fut_traj, obs_traj_vel, fut_traj_vel, seq_start_end, obs_frames, fut_frames, past_obst, fut_obst)  = next(iterator)
            state = torch.cat([obs_traj, fut_traj], dim=0)
            state = state.view(-1, state.shape[2])
            map = torch.cat([past_obst, fut_obst], dim=0)
            map = map.view(-1, map.shape[2], map.shape[3], map.shape[4])

            obst_feat = self.encoder(state, map, train=True)



            # 첫번째 iteration 디코더 인풋 = (obs_traj_vel의 마지막 값, (hidden_state, cell_state))
            # where hidden_state = "인코더의 마지막 hidden_layer아웃풋과 그것으로 만든 max_pooled값을 concat해서 mlp 통과시켜만든 feature인 noise_input에다 noise까지 추가한값)"
            recon_map, pred_vel = self.decoder(
                obst_feat
            )

            # focal_loss = self.alpha * map * torch.log(recon_map + self.eps) * ((1-recon_map) ** self.gamma) \
            #              + (1-self.alpha) * (1 - map) * torch.log(1 - recon_map + self.eps) * (recon_map ** self.gamma)

            map_loss = - (torch.log(recon_map + self.eps) * map +
              torch.log(1 - recon_map + self.eps) * (1 - map))

            recon_vel = F.mse_loss(pred_vel, state[:,2:4], reduction='sum')

            loss =  map_loss.sum().div(state.shape[0]) + recon_vel.div(state.shape[0])

            self.optim_vae.zero_grad()
            loss.backward()
            self.optim_vae.step()


            # save model parameters
            if iteration % self.ckpt_save_iter == 0:
                self.save_checkpoint(iteration)


            # (visdom) insert current line stats
            if self.viz_on and (iteration % self.viz_ll_iter == 0):
                test_loss = self.test()
                self.line_gather.insert(iter=iteration,
                                        loss=loss.item(),
                                        test_loss=test_loss.item(),
                                        )
                prn_str = ('[iter_%d (epoch_%d)] loss: %.3f \n'
                          ) % \
                          (iteration, epoch,
                           loss.item())

                print(prn_str)
                if self.record_file:
                    record = open(self.record_file, 'a')
                    record.write('%s\n' % (prn_str,))
                    record.close()


            # (visdom) visualize line stats (then flush out)
            if self.viz_on and (iteration % self.viz_la_iter == 0):
                self.visualize_line()
                self.line_gather.flush()
                # self.recon(self.train_loader)
                # self.recon(self.val_loader)

    def test(self):
        self.set_mode(train=False)
        loss=0
        b = 0
        with torch.no_grad():
            for abatch in self.val_loader:
                b += 1

                (obs_traj, fut_traj, obs_traj_vel, fut_traj_vel, seq_start_end, obs_frames, fut_frames, past_obst,
                 fut_obst) = abatch
                batch = fut_traj.size(1)

                state = torch.cat([obs_traj, fut_traj], dim=0)
                state = state.view(-1, state.shape[2])
                map = torch.cat([past_obst, fut_obst], dim=0)
                map = map.view(-1, map.shape[2], map.shape[3], map.shape[4])

                obst_feat = self.encoder(state, map, train=True)

                recon_map, pred_vel = self.decoder(
                    obst_feat
                )


                map_loss = - (torch.log(recon_map + self.eps) * map +
                              torch.log(1 - recon_map + self.eps) * (1 - map))

                recon_vel = F.mse_loss(pred_vel, state[:, 2:4], reduction='sum')

                loss += (map_loss.sum().div(state.shape[0]) + recon_vel.div(state.shape[0]))

        self.set_mode(train=True)
        return loss.div(b)

    ####

    def recon(self):
        self.set_mode(train=False)
        with torch.no_grad():
            # if 'eth' in self.name:
            fixed_idxs = [115,16, 53, 224]
            # fixed_idxs = range(30,60)
            # fixed_idxs = range(49)
            dset='test'
            data_loader = self.val_loader

##########################
            data = []
            for i, idx in enumerate(fixed_idxs):
                data.append(data_loader.dataset.__getitem__(idx))


            (obs_traj, fut_traj, obs_traj_vel, fut_traj_vel, seq_start_end, obs_frames, fut_frames, past_obst,
             fut_obst) = seq_collate(data)
            # out_dir = os.path.join('./output',self.name, dset + str(self.ckpt_load_iter))
            # mkdirs(out_dir)
            # for i in range(fut_obst.shape[1]):
            #     save_image(fut_obst[:, i], str(os.path.join(out_dir, 'gt_img'+str(i)+'.png')), nrow=self.pred_len, pad_value=1)


            state = obs_traj[0]
            map = past_obst[0]

            obst_feat = self.encoder(state, map)

            recon_map, _ = self.decoder(
                obst_feat
            )
            for i in range(map.shape[0]):
                print(i, recon_map[i].max().item(
                ))

            out_dir = os.path.join('./output',self.name, dset)
            mkdirs(out_dir)
            for i in range(map.shape[0]):
                save_image(recon_map[i], str(os.path.join(out_dir, 'recon_img'+str(i)+'.png')), nrow=self.pred_len, pad_value=1)
                save_image(map[i], str(os.path.join(out_dir, 'gt_img'+str(i)+'.png')), nrow=self.pred_len, pad_value=1)

        self.set_mode(train=True)




    ####
    def viz_init(self):
        self.viz.close(env=self.name + '/lines', win=self.win_id['test_loss'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['loss'])

    ####
    def visualize_line(self):

        # prepare data to plot
        data = self.line_gather.data
        iters = torch.Tensor(data['iter'])
        test_loss = torch.Tensor(data['test_loss'])
        loss = torch.Tensor(data['loss'])

        self.viz.line(
            X=iters, Y=loss, env=self.name + '/lines',
            win=self.win_id['loss'], update='append',
            opts=dict(xlabel='iter', ylabel='loss',
                      title='Recon. loss')
        )

        self.viz.line(
            X=iters, Y=test_loss, env=self.name + '/lines',
            win=self.win_id['test_loss'], update='append',
            opts=dict(xlabel='iter', ylabel='test_loss',
                      title='Recon. loss - Test'),
        )



    def set_mode(self, train=True):

        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

    ####
    def save_checkpoint(self, iteration):

        encoder_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoder.pt' % iteration
        )
        decoder_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_decoder.pt' % iteration
        )


        mkdirs(self.ckpt_dir)

        torch.save(self.encoder, encoder_path)
        torch.save(self.decoder, decoder_path)
    ####
    def load_checkpoint(self):

        encoder_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoder.pt' % self.ckpt_load_iter
        )
        decoder_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_decoder.pt' % self.ckpt_load_iter
        )

        if self.device == 'cuda':
            self.encoder = torch.load(encoder_path)
            self.decoder = torch.load(decoder_path)
        else:
            self.encoder = torch.load(encoder_path, map_location='cpu')
            self.decoder = torch.load(decoder_path, map_location='cpu')

    def load_map_weights(self, map_path):
        if self.device == 'cuda':
            loaded_map_w = torch.load(map_path)
        else:
            loaded_map_w = torch.load(map_path, map_location='cpu')
        self.encoder.conv1.weight = loaded_map_w.map_net.conv1.weight
        self.encoder.conv2.weight = loaded_map_w.map_net.conv2.weight
        self.encoder.conv3.weight = loaded_map_w.map_net.conv3.weight