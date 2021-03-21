import os

import torch.optim as optim
# -----------------------------------------------------------------------------#
from utils import DataGather, mkdirs, grid2gif2, apply_poe, sample_gaussian, sample_gumbel_softmax
from model import *
from loss import kl_two_gaussian, displacement_error, final_displacement_error
from utils_sgan import relative_to_abs, get_dset_path
from data.loader import data_loader
from eval_util import ploot

import matplotlib.pyplot as plt
from torch.distributions import RelaxedOneHotCategorical as concrete
from torch.distributions import OneHotCategorical as discrete
from torch.distributions import kl_divergence

###############################################################################

class Solver(object):

    ####
    def __init__(self, args):

        self.args = args

        # self.name = '%s_pred_len_%s_zS_%s_lr_%s_embedding_dim_%s_encoder_h_dim_%s_mlp_dim_%s_pool_dim_%s' % \
        #             (args.dataset_name, args.pred_len, args.zS_dim, args.embedding_dim, args.encoder_h_dim, args.mlp_dim, args.pool_dim, args.lr_VAE)
        self.name = '%s_pred_len_%s_zS_%s_embedding_dim_%s_enc_h_dim_%s_dec_h_dim_%s_mlp_dim_%s_pool_dim_%s_lr_%s_klw_%s' % \
                    (args.dataset_name, args.pred_len, args.zS_dim, args.embedding_dim, args.encoder_h_dim, args.decoder_h_dim, args.mlp_dim, args.pool_dim, args.lr_VAE, args.kl_weight)
        # to be appended by run_id

        # self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = args.device
        self.temp=1.99
        self.kl_weight=1.0

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
                recon='win_recon', loss_kl='win_loss_kl', loss_recon='win_loss_recon', total_loss='win_total_loss'
                , ade_min='win_ade_min', fde_min='win_fde_min', ade_avg='win_ade_avg', fde_avg='win_fde_avg',
                ade_std='win_ade_std', fde_std='win_fde_std',
                test_loss_recon='win_test_loss_recon', test_loss_kl='win_test_loss_kl', test_total_loss='win_test_total_loss'
            )
            self.line_gather = DataGather(
                'iter', 'loss_recon', 'loss_kl', 'total_loss', 'ade_min', 'fde_min',
                'ade_avg', 'fde_avg', 'ade_std', 'fde_std',
                'test_loss_recon', 'test_loss_kl', 'test_total_loss'
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

        if args.pool_dim==0:
            pooling_type=None
        else:
            pooling_type=args.pooling_type

        if self.ckpt_load_iter == 0 or args.dataset_name =='all':  # create a new model
            self.encoderMx = Encoder(
                args.zS_dim,
                embedding_dim=args.embedding_dim,
                enc_h_dim=args.encoder_h_dim,
                mlp_dim=args.mlp_dim,
                pool_dim=args.pool_dim,
                batch_norm=args.batch_norm,
                num_layers=args.num_layers,
                dropout=args.dropout,
                pooling_type=pooling_type).to(self.device)
            self.encoderMy = Encoder(
                args.zS_dim,
                embedding_dim=args.embedding_dim,
                enc_h_dim=args.encoder_h_dim,
                mlp_dim=args.mlp_dim,
                pool_dim=args.pool_dim,
                batch_norm=args.batch_norm,
                num_layers=args.num_layers,
                dropout=args.dropout,
                pooling_type=pooling_type,
                coditioned=True).to(self.device)
            self.decoderMy = Decoder(
                args.pred_len,
                embedding_dim=args.embedding_dim,
                dec_h_dim=self.decoder_h_dim,
                enc_h_dim=args.encoder_h_dim,
                z_dim=args.zS_dim,
                mlp_dim=args.mlp_dim,
                num_layers=args.num_layers,
                device=args.device,
                dropout=args.dropout,
                pool_dim=args.pool_dim,
                batch_norm=args.batch_norm).to(self.device)

        else:  # load a previously saved model
            print('Loading saved models (iter: %d)...' % self.ckpt_load_iter)
            self.load_checkpoint()
            print('...done')


        # get VAE parameters
        vae_params = \
            list(self.encoderMx.parameters()) + \
            list(self.encoderMy.parameters()) + \
            list(self.decoderMy.parameters())

        # create optimizers
        self.optim_vae = optim.Adam(
            vae_params,
            lr=self.lr_VAE,
            betas=[self.beta1_VAE, self.beta2_VAE]
        )

        # prepare dataloader (iterable)
        print('Start loading data...')
        train_path = os.path.join(self.dataset_dir, self.dataset_name, 'train')
        val_path = os.path.join(self.dataset_dir, self.dataset_name, 'test')

        # long_dtype, float_dtype = get_dtypes(args)

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
            (obs_traj, fut_traj, obs_traj_rel, fut_traj_rel, non_linear_ped,
             loss_mask, seq_start_end, obs_frames, pred_frames) = next(iterator)
            batch = obs_traj_rel.size(1) #=sum(seq_start_end[:,1] - seq_start_end[:,0])


            (dist_fc_inputMx, logitX) \
                = self.encoderMx(obs_traj_rel, seq_start_end)
            (dist_fc_inputMy, logitY) \
                = self.encoderMy(fut_traj_rel, seq_start_end, coditioned_h=dist_fc_inputMx)

            p_dist = discrete(logits=logitX)
            q_dist = discrete(logits=logitY)
            relaxed_q_dist = concrete(logits=logitY, temperature=self.temp)

            last_pos = obs_traj[-1]  # (batchsize, 2)
            last_pos_rel = obs_traj_rel[-1]  # (batchsize, 2)
            # Predict Trajectory

            # 첫번째 iteration 디코더 인풋 = (obs_traj_rel의 마지막 값, (hidden_state, cell_state))
            # where hidden_state = "인코더의 마지막 hidden_layer아웃풋과 그것으로 만든 max_pooled값을 concat해서 mlp 통과시켜만든 feature인 noise_input에다 noise까지 추가한값)"
            pred_fut_traj_rel = self.decoderMy(
                last_pos,
                last_pos_rel,
                dist_fc_inputMx,
                relaxed_q_dist.rsample(),
                seq_start_end
            )
            pred_fut_traj = relative_to_abs(
                pred_fut_traj_rel, obs_traj[-1]
            )

            ################## total loss for vae ####################
            loss_recon = F.mse_loss(pred_fut_traj, fut_traj, reduction='sum').div(batch)
            # test
            loss_kl = kl_divergence(q_dist, p_dist).sum().div(batch)
            loss_kl = torch.clamp(loss_kl, min=0.07)
            vae_loss = loss_recon + self.kl_weight * loss_kl

            #### dist loss ####
            # pred_fut_traj = relative_to_abs(
            #     pred_fut_traj_rel, obs_traj[-1]
            # )
            # for _, (start, end) in enumerate(seq_start_end):
            #     start = start.item()
            #     end = end.item()
            #     num_ped = end - start
            #     one_frame_slide = pred_fut_traj[:, start:end, :]  # (pred_len, num_ped, 2)
            #     skip_idx=[num_ped*idx + idx for idx in range(num_ped)]
            #     for i in range(self.pred_len):
            #         curr_frame = one_frame_slide[i]  # frame of time=i #(num_ped,2)
            #         curr1 = curr_frame.repeat(num_ped, 1)
            #         curr2 = self.repeat(curr_frame, num_ped)
            #         dist = torch.sqrt(torch.pow(curr1 - curr2, 2).sum(1))
            #         inv_dist = 1 / dist[[idx for idx in range(len(dist)) if idx not in skip_idx]]
            #         dist = torch.inverse(dist)
            #         torch.sigmoid()


            self.optim_vae.zero_grad()
            vae_loss.backward()
            self.optim_vae.step()


            # print the losses
            # if iteration % self.print_iter == 0:


            # save model parameters
            if iteration % self.ckpt_save_iter == 0:
                self.save_checkpoint(iteration)

            # save output images (recon, synth, etc.)
            # if iteration % self.eval_metrics_iter == 0:

            # (visdom) insert current line stats
            if self.viz_on and (iteration % self.viz_ll_iter == 0):
                ade_min, fde_min, _, _, \
                ade_avg, fde_avg, _, _, \
                ade_std, fde_std, _, _, \
                test_loss_recon, test_loss_kl, test_vae_loss = self.evaluate_dist_collision(self.val_loader, 20, 0.1, loss=True)
                self.line_gather.insert(iter=iteration,
                                        loss_recon=loss_recon.item(),
                                        loss_kl=loss_kl.item(),
                                        total_loss=vae_loss.item(),
                                        ade_min=ade_min,
                                        fde_min=fde_min,
                                        ade_avg=ade_avg,
                                        fde_avg=fde_avg,
                                        ade_std=ade_std,
                                        fde_std=fde_std,
                                        test_loss_recon=test_loss_recon.item(),
                                        test_loss_kl=test_loss_kl.item(),
                                        test_total_loss=test_vae_loss.item(),
                                        )
                prn_str = ('[iter_%d (epoch_%d)] vae_loss: %.3f ' + \
                              '(recon: %.3f, kl: %.3f)\n' + \
                              'ADE min: %.2f, FDE min: %.2f, ADE avg: %.2f, FDE avg: %.2f\n'
                          ) % \
                          (iteration, epoch,
                           vae_loss.item(), loss_recon.item(), loss_kl.item(),
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

    def evaluate(self, num_samples, data_loader):
        self.set_mode(train=False)
        ade_outer, fde_outer = [], []
        total_traj = 0
        with torch.no_grad():
            for batch in data_loader:
                (obs_traj, fut_traj, obs_traj_rel, fut_traj_rel, non_linear_ped,
                 loss_mask, seq_start_end, obs_frames, pred_frames) = batch
                ade, fde = [], []
                total_traj += fut_traj.size(1)

                for _ in range(num_samples):
                    (dist_fc_inputMx, muSharedMx, stdSharedMx) \
                        = self.encoderMx(obs_traj_rel, seq_start_end)
                    zSharedMx = sample_gaussian(self.device, muSharedMx, stdSharedMx)
                    decoder_h = torch.cat([dist_fc_inputMx, zSharedMx], dim=1).unsqueeze(0)
                    decoder_c = torch.zeros(self.num_layers, obs_traj.size(1), self.decoder_h_dim).to(self.device)

                    pred_fut_traj_rel = self.decoderMy(
                        obs_traj[-1],
                        obs_traj_rel[-1],
                        (decoder_h, decoder_c),
                        seq_start_end,
                    )

                    pred_fut_traj = relative_to_abs(
                        pred_fut_traj_rel, obs_traj[-1]
                    )
                    ade.append(displacement_error(
                        pred_fut_traj, fut_traj, mode='raw'
                    ))
                    fde.append(final_displacement_error(
                        pred_fut_traj[-1], fut_traj[-1], mode='raw'
                    ))

                ade_sum, _, _ = self.evaluate_helper(ade, seq_start_end)
                fde_sum, _, _ = self.evaluate_helper(fde, seq_start_end)

                ade_outer.append(ade_sum)
                fde_outer.append(fde_sum)
            ade = sum(ade_outer) / (total_traj * self.pred_len)
            fde = sum(fde_outer) / (total_traj)
        self.set_mode(train=True)
        return ade, fde


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


    def evaluate_dist(self, data_loader, num_samples, threshold):
        self.set_mode(train=False)
        ade_outer_min, fde_outer_min = [], []
        ade_outer_avg, fde_outer_avg = [], []
        ade_outer_std, fde_outer_std = [], []

        coll_rate_outer_min, coll_rate_outer_avg, coll_rate_outer_std = [], [], []
        total_traj = 0
        with torch.no_grad():
            for batch in data_loader:
                (obs_traj, fut_traj, obs_traj_rel, fut_traj_rel, non_linear_ped,
                 loss_mask, seq_start_end, obs_frames, pred_frames) = batch
                ade, fde = [], []
                total_traj += fut_traj.size(1)

                coll_rate = []
                (dist_fc_inputMx, muSharedMx, stdSharedMx) \
                    = self.encoderMx(obs_traj_rel, seq_start_end)
                for _ in range(num_samples):
                    zSharedMx = sample_gaussian(self.device, muSharedMx, stdSharedMx)
                    decoder_h = torch.cat([dist_fc_inputMx, zSharedMx], dim=1).unsqueeze(0)
                    decoder_c = torch.zeros(self.num_layers, obs_traj.size(1), self.decoder_h_dim).to(self.device)

                    pred_fut_traj_rel = self.decoderMy(
                        obs_traj[-1],
                        obs_traj_rel[-1],
                        (decoder_h, decoder_c),
                        seq_start_end,
                    )
                    pred_fut_traj = relative_to_abs(
                        pred_fut_traj_rel, obs_traj[-1]
                    )
                    ade.append(displacement_error(
                        pred_fut_traj, fut_traj, mode='raw'
                    ))
                    fde.append(final_displacement_error(
                        pred_fut_traj[-1], fut_traj[-1], mode='raw'
                    ))
                    n_pred_frame = 0.
                    n_collision = 0.

                    for _, (start, end) in enumerate(seq_start_end):
                        start = start.item()
                        end = end.item()
                        num_ped = end - start
                        one_frame_slide = pred_fut_traj[:,start:end,:] # (pred_len, num_ped, 2)
                        for i in range(self.pred_len):
                            n_pred_frame +=1
                            curr_frame = one_frame_slide[i] # frame of time=i #(num_ped,2)
                            curr1 = curr_frame.repeat(num_ped, 1)
                            curr2 = self.repeat(curr_frame, num_ped)
                            dist = torch.sqrt(torch.pow(curr1 - curr2, 2).sum(1))
                            # dist = dist.reshape(num_ped, num_ped).numpy()
                            # diff_agent_dist = dist[np.triu_indices(4, k=1)]

                            if (dist < threshold).sum() > num_ped:
                                n_collision +=1

                            # for dist_idx in combinations(range(num_ped), 2):
                            #     two_points = curr_frame[list(dist_idx)]
                            #     dist = torch.pow(two_points[0] - two_points[1], 2).sum()
                            #     if dist < threshold:
                            #         n_collision +=1
                            #         break
                    coll_rate.append(torch.tensor(n_collision / n_pred_frame))

                ade_sum_min, ade_sum_avg, ade_sum_std = self.evaluate_helper(ade, seq_start_end)
                fde_sum_min, fde_sum_avg, fde_sum_std = self.evaluate_helper(fde, seq_start_end)
                ade_outer_min.append(ade_sum_min)
                fde_outer_min.append(fde_sum_min)
                ade_outer_avg.append(ade_sum_avg)
                fde_outer_avg.append(fde_sum_avg)
                ade_outer_std.append(ade_sum_std)
                fde_outer_std.append(fde_sum_std)

                coll_rate = np.array(coll_rate)
                coll_rate_outer_min.append(coll_rate.min())
                coll_rate_outer_avg.append(coll_rate.mean())
                coll_rate_outer_std.append(coll_rate.std())
            ade_min = sum(ade_outer_min) / (total_traj * self.pred_len)
            fde_min = sum(fde_outer_min) / (total_traj)
            ade_avg = sum(ade_outer_avg) / (total_traj * self.pred_len)
            fde_avg = sum(fde_outer_avg) / (total_traj)
            ade_std = sum(ade_outer_std) / (total_traj * self.pred_len)
            fde_std = sum(fde_outer_std) / (total_traj)
            coll_rate_min = sum(coll_rate_outer_min) / len(coll_rate_outer_min) * 100
            coll_rate_avg = sum(coll_rate_outer_avg) / len(coll_rate_outer_avg) * 100
            coll_rate_std = sum(coll_rate_outer_std) / len(coll_rate_outer_std) * 100
        self.set_mode(train=True)
        return ade_min, fde_min, coll_rate_min, \
               ade_avg, fde_avg, coll_rate_avg, \
               ade_std, fde_std, coll_rate_std

    def evaluate_dist_collision(self, data_loader, num_samples, threshold, loss=False):
        self.set_mode(train=False)
        ade_outer_min, fde_outer_min = [], []
        ade_outer_avg, fde_outer_avg = [], []
        ade_outer_std, fde_outer_std = [], []
        total_traj = 0
        all_coll = []
        all_ade, all_fde = [], []
        all_ade_stat, all_fde_stat = [], []

        loss_recon = loss_kl = vae_loss = 0

        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_rel, fut_traj_rel, non_linear_ped,
                 loss_mask, seq_start_end, obs_frames, pred_frames) = batch
                batch_size = obs_traj_rel.size(1)
                ade, fde = [], []
                total_traj += fut_traj.size(1)

                (dist_fc_inputMx, logitX) \
                    = self.encoderMx(obs_traj_rel, seq_start_end)
                p_dist = discrete(logits=logitX)
                relaxed_p_dist = concrete(logits=logitX, temperature=self.temp)
                if loss:
                    (dist_fc_inputMy, logitY) \
                        = self.encoderMy(fut_traj_rel, seq_start_end, coditioned_h=dist_fc_inputMx)

                    q_dist = discrete(logits=logitY)
                    last_pos = obs_traj[-1]  # (batchsize, 2)
                    last_pos_rel = obs_traj_rel[-1]  # (batchsize, 2)
                    # Predict Trajectory

                    # 첫번째 iteration 디코더 인풋 = (obs_traj_rel의 마지막 값, (hidden_state, cell_state))
                    # where hidden_state = "인코더의 마지막 hidden_layer아웃풋과 그것으로 만든 max_pooled값을 concat해서 mlp 통과시켜만든 feature인 noise_input에다 noise까지 추가한값)"
                    pred_fut_traj_rel = self.decoderMy(
                        last_pos,
                        last_pos_rel,
                        dist_fc_inputMx,
                        relaxed_p_dist.rsample(),
                        seq_start_end
                    )
                    pred_fut_traj = relative_to_abs(
                        pred_fut_traj_rel, obs_traj[-1]
                    )

                    loss_recon += F.mse_loss(pred_fut_traj, fut_traj_rel, reduction='sum').div(batch_size)
                    loss_kl = kl_divergence(q_dist, p_dist).sum().div(batch_size)
                    loss_kl = torch.clamp(loss_kl, min=0.07)
                    vae_loss += (loss_recon + self.kl_weight * loss_kl)



                coll_20samples = [] # (20, # seq, 12)
                for _ in range(num_samples):
                    pred_fut_traj_rel = self.decoderMy(
                        obs_traj[-1],
                        obs_traj_rel[-1],
                        dist_fc_inputMx,
                        relaxed_p_dist.rsample(),
                        seq_start_end,
                    )
                    pred_fut_traj = relative_to_abs(
                        pred_fut_traj_rel, obs_traj[-1]
                    )
                    ade.append(displacement_error(
                        pred_fut_traj, fut_traj, mode='raw'
                    ))
                    fde.append(final_displacement_error(
                        pred_fut_traj[-1], fut_traj[-1], mode='raw'
                    ))

                    seq_coll = [] #64
                    for idx, (start, end) in enumerate(seq_start_end):

                        start = start.item()
                        end = end.item()
                        num_ped = end - start
                        one_frame_slide = pred_fut_traj[:,start:end,:] # (pred_len, num_ped, 2)

                        frame_coll = [] #num_ped
                        for i in range(self.pred_len):
                            curr_frame = one_frame_slide[i] # frame of time=i #(num_ped,2)
                            curr1 = curr_frame.repeat(num_ped, 1)
                            curr2 = self.repeat(curr_frame, num_ped)
                            dist = torch.sqrt(torch.pow(curr1 - curr2, 2).sum(1)).cpu().numpy()
                            dist = dist.reshape(num_ped, num_ped)
                            diff_agent_idx = np.triu_indices(num_ped, k=1)
                            diff_agent_dist = dist[diff_agent_idx]
                            curr_coll_rate = (diff_agent_dist < threshold).sum() / len(diff_agent_dist)
                            # if (diff_agent_dist < threshold).sum() > 0:
                            #     print(idx)
                            #     print(diff_agent_dist)
                            #     print('---------------------')
                            frame_coll.append(curr_coll_rate)
                        seq_coll.append(frame_coll)

                    coll_20samples.append(seq_coll)

                ade_sum_min, ade_sum_avg, ade_sum_std = self.evaluate_helper(ade, seq_start_end)
                fde_sum_min, fde_sum_avg, fde_sum_std = self.evaluate_helper(fde, seq_start_end)
                ade_outer_min.append(ade_sum_min)
                fde_outer_min.append(fde_sum_min)
                ade_outer_avg.append(ade_sum_avg)
                fde_outer_avg.append(fde_sum_avg)
                ade_outer_std.append(ade_sum_std)
                fde_outer_std.append(fde_sum_std)

                all_ade.append(torch.stack(ade, dim=1).cpu().numpy())
                all_fde.append(torch.stack(fde, dim=1).cpu().numpy())
                all_ade_stat.append(self.evaluate_helper2(ade, seq_start_end))
                all_fde_stat.append(self.evaluate_helper2(fde, seq_start_end))

                all_coll.append(np.array(coll_20samples))

            all_coll=np.concatenate(all_coll, axis=1) #(20,70,12)
            coll_rate_min=all_coll.min(axis=0).mean()*100
            coll_rate_avg=all_coll.mean(axis=0).mean()*100
            coll_rate_std=all_coll.std(axis=0).mean()*100

            all_ade = np.concatenate(all_ade, axis=0)
            all_fde = np.concatenate(all_fde, axis=0)
            all_ade_stat=np.concatenate(all_ade_stat, axis=0)
            all_fde_stat=np.concatenate(all_fde_stat, axis=0)
            # all_ade_stat=np.concatenate(all_ade_stat, axis=1) / total_traj
            # all_fde_stat=np.concatenate(all_fde_stat, axis=1) / total_traj
            import pandas as pd
            pd.DataFrame(all_ade).to_csv("./ade_" +self.dataset_name+ ".csv")
            pd.DataFrame(all_fde).to_csv("./fde_" +self.dataset_name+ ".csv")
            pd.DataFrame(all_ade_stat).to_csv("./ade_seq_stat_divided_" +self.dataset_name+ ".csv")
            pd.DataFrame(all_fde_stat).to_csv("./fde_seq_stat_divided_" +self.dataset_name+ ".csv")

            #non-zero coll
            non_zero_coll_avg = []
            non_zero_coll_min = []
            non_zero_coll_std = []
            for sample in all_coll: #sample = [70,12]
                non_zero_idx = np.where(sample > 0)
                if len(non_zero_idx[0]) > 0:
                    non_zero_coll_avg.append(sample[non_zero_idx].mean())
                    non_zero_coll_std.append(sample[non_zero_idx].std())
                    non_zero_coll_min.append(sample[non_zero_idx].min())

            non_zero_coll_avg = np.array(non_zero_coll_avg).mean()*100
            non_zero_coll_min = np.array(non_zero_coll_min).mean() *100
            non_zero_coll_std = np.array(non_zero_coll_std).mean() *100

            ade_min = sum(ade_outer_min) / (total_traj * self.pred_len)
            fde_min = sum(fde_outer_min) / (total_traj)
            ade_avg = sum(ade_outer_avg) / (total_traj * self.pred_len)
            fde_avg = sum(fde_outer_avg) / (total_traj)
            ade_std = sum(ade_outer_std) / (total_traj * self.pred_len)
            fde_std = sum(fde_outer_std) / (total_traj)
        self.set_mode(train=True)
        if loss:
            return ade_min, fde_min, coll_rate_min, non_zero_coll_min, \
                   ade_avg, fde_avg, coll_rate_avg, non_zero_coll_avg, \
                   ade_std, fde_std, coll_rate_std, non_zero_coll_std, \
                   loss_recon/b, loss_kl/b, vae_loss/b
        else:
            return ade_min, fde_min, coll_rate_min, non_zero_coll_min, \
                   ade_avg, fde_avg, coll_rate_avg, non_zero_coll_avg, \
                   ade_std, fde_std, coll_rate_std, non_zero_coll_std


    def plot_traj(self, data_loader):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        import cv2
        # read video
        cap = cv2.VideoCapture('D:\crowd\ewap_dataset\seq_eth\seq_eth.avi')
        frame_number=880
        cap.set(1, frame_number)
        ret, frame = cap.read()

        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111)
        # ax.imshow(frame)

        h = np.loadtxt('D:\crowd\ewap_dataset\seq_eth\H.txt')
        inv_h_t = np.linalg.pinv(np.transpose(h))

        total_traj = 0
        b = 0
        with torch.no_grad():
            for batch in data_loader:
                b +=1
                (obs_traj, fut_traj, obs_traj_rel, fut_traj_rel, non_linear_ped,
                 loss_mask, seq_start_end, obs_frames, pred_frames) = batch
                total_traj += fut_traj.size(1)

                (dist_fc_inputMx, muSharedMx, stdSharedMx) \
                    = self.encoderMx(obs_traj_rel, seq_start_end)

                zSharedMx = sample_gaussian(self.device, muSharedMx, stdSharedMx)
                decoder_h = torch.cat([dist_fc_inputMx, zSharedMx], dim=1).unsqueeze(0)
                decoder_c = torch.zeros(self.num_layers, obs_traj.size(1), self.decoder_h_dim).to(self.device)

                pred_fut_traj_rel = self.decoderMy(
                    obs_traj[-1],
                    obs_traj_rel[-1],
                    (decoder_h, decoder_c),
                    seq_start_end,
                )
                pred_fut_traj = relative_to_abs(
                    pred_fut_traj_rel, obs_traj[-1]
                )




                frame_numbers = [10330, 10340, 10390, 12020, 12100]
                frame_number = frame_numbers[4]
                cap.set(1, frame_number)
                ret, frame = cap.read()
                frmae_seq_idx = np.where(pred_frames[:,0] == frame_number)[0]
                rng = range(frmae_seq_idx[0], frmae_seq_idx[-1]+1)



                gt_data, pred_data = [], []

                for idx in range(len(rng)):
                    one_ped = rng[idx]
                    obs_real = obs_traj[:, one_ped]
                    obs_real = np.concatenate([obs_real, np.ones((self.obs_len, 1))], axis=1)
                    obs_pixel = np.matmul(obs_real, inv_h_t)
                    obs_pixel /= np.expand_dims(obs_pixel[:, 2], 1)

                    gt_real = fut_traj[:, one_ped]
                    gt_real = np.concatenate([gt_real, np.ones((self.pred_len, 1))], axis=1)
                    gt_pixel = np.matmul(gt_real, inv_h_t)
                    gt_pixel /= np.expand_dims(gt_pixel[:, 2], 1)

                    pred_real = pred_fut_traj[:, one_ped].numpy()
                    pred_pixel = np.concatenate([pred_real, np.ones((self.pred_len, 1))], axis=1)
                    pred_pixel = np.matmul(pred_pixel, inv_h_t)
                    pred_pixel /= np.expand_dims(pred_pixel[:, 2], 1)

                    gt_data.append(np.concatenate([obs_pixel, gt_pixel], 0)) # (20, 3)
                    pred_data.append(np.concatenate([obs_pixel, pred_pixel], 0))

                gt_data = np.stack(gt_data)
                pred_data = np.stack(pred_data)

                if self.dataset_name == 'eth':
                    gt_data[:,:, [0,1]] = gt_data[:,:,[1,0]]
                    pred_data[:,:,[0,1]] = pred_data[:,:,[1,0]]


                n_agent = gt_data.shape[0]
                n_frame = gt_data.shape[1]

                fig, ax = plt.subplots()

                ln_gt = []
                ln_pred = []
                # colors = ['red', 'magenta', 'lightgreen', 'slateblue', 'blue', 'darkgreen', 'darkorange',
                #      'gray', 'purple', 'turquoise', 'midnightblue', 'olive', 'black', 'pink', 'burlywood', 'yellow']

                colors = ['r', 'g', 'y', 'm', 'c', 'k', 'w', 'b']
                for i in range(n_agent):
                    ln_gt.append(ax.plot([], [], colors[i] + '--')[0])
                    ln_pred.append(ax.plot([], [], colors[i] + ':')[0])


                def init():
                    ax.imshow(frame)
                    # ax.set_xlim(-10, 15)
                    # ax.set_ylim(-10, 15)

                def update_dot(num_t):
                    print(num_t)
                    # if (num_t < n_frame):
                    for i in range(n_agent):
                        ln_gt[i].set_data(gt_data[i, :num_t, 0], gt_data[i, :num_t, 1])
                        ln_pred[i].set_data(pred_data[i, :num_t, 0][:num_t], pred_data[i, :num_t, 1])

                ani = FuncAnimation(fig, update_dot, frames=n_frame, interval=100, init_func=init())

                writer = PillowWriter(fps=60)
                # ani.save("eth_frame10360_rng106to110.gif", writer=writer)
                ani.save("eth_frame" + str(frame_number) + "_rng" + str(rng[0]) +"to" +str(rng[-1]) +".gif", writer=writer)
                print('---------------')
                plt.close()

                ploot(gt_data, pred_data, frame, b)


    def plot_traj_var(self, data_loader):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        import cv2
        # read video
        cap = cv2.VideoCapture('D:\crowd\ewap_dataset\seq_eth\seq_eth.avi')

        colors = ['r', 'g', 'y', 'm', 'c', 'k', 'w', 'b']
        h = np.loadtxt('D:\crowd\ewap_dataset\seq_eth\H.txt')
        inv_h_t = np.linalg.pinv(np.transpose(h))

        total_traj = 0
        b = 0
        with torch.no_grad():
            for batch in data_loader:
                b +=1
                (obs_traj, fut_traj, obs_traj_rel, fut_traj_rel, non_linear_ped,
                 loss_mask, seq_start_end, obs_frames, pred_frames) = batch
                total_traj += fut_traj.size(1)

                (dist_fc_inputMx, logitX) \
                    = self.encoderMx(obs_traj_rel, seq_start_end)
                p_dist = discrete(logits=logitX)
###########
                # frame_numbers = [10330, 10340, 10390, 12020, 12100]
                # frame_number = frame_numbers[4]
                # cap.set(1, frame_number)
                # ret, frame = cap.read()
                # frmae_seq_idx = np.where(pred_frames[:,0] == frame_number)[0]
                # rng = range(frmae_seq_idx[0], frmae_seq_idx[-1]+1)



                agent_rng = range(146, 149)
                # frame_number = obs_frames[95][-1]
                frame_numbers = np.concatenate([obs_frames[agent_rng[0]], pred_frames[agent_rng[0]]])
                frame_number = frame_numbers[0]
                cap.set(1, frame_number)
                ret, frame = cap.read()
                multi_sample_pred = []

                # (dist_fc_inputMy, muSharedMy, stdSharedMy) \
                #     = self.encoderMy(fut_traj_rel, seq_start_end, coditioned_h=dist_fc_inputMx)

                for _ in range(20):
                    zSharedMx = p_dist.rsample()
                    decoder_h = torch.cat([dist_fc_inputMx, zSharedMx], dim=1).unsqueeze(0)
                    decoder_c = torch.zeros(self.num_layers, obs_traj.size(1), self.decoder_h_dim).to(self.device)

                    pred_fut_traj_rel = self.decoderMy(
                        obs_traj[-1],
                        obs_traj_rel[-1],
                        (decoder_h, decoder_c),
                        seq_start_end,
                    )
                    pred_fut_traj = relative_to_abs(
                        pred_fut_traj_rel, obs_traj[-1]
                    )


                    gt_data, pred_data = [], []

                    for idx in range(len(agent_rng)):
                        one_ped = agent_rng[idx]
                        obs_real = obs_traj[:, one_ped]
                        obs_real = np.concatenate([obs_real, np.ones((self.obs_len, 1))], axis=1)
                        obs_pixel = np.matmul(obs_real, inv_h_t)
                        obs_pixel /= np.expand_dims(obs_pixel[:, 2], 1)

                        gt_real = fut_traj[:, one_ped]
                        gt_real = np.concatenate([gt_real, np.ones((self.pred_len, 1))], axis=1)
                        gt_pixel = np.matmul(gt_real, inv_h_t)
                        gt_pixel /= np.expand_dims(gt_pixel[:, 2], 1)

                        pred_real = pred_fut_traj[:, one_ped].numpy()
                        pred_pixel = np.concatenate([pred_real, np.ones((self.pred_len, 1))], axis=1)
                        pred_pixel = np.matmul(pred_pixel, inv_h_t)
                        pred_pixel /= np.expand_dims(pred_pixel[:, 2], 1)

                        gt_data.append(np.concatenate([obs_pixel, gt_pixel], 0)) # (20, 3)
                        pred_data.append(np.concatenate([obs_pixel, pred_pixel], 0))

                    gt_data = np.stack(gt_data)
                    pred_data = np.stack(pred_data)

                    if self.dataset_name == 'eth':
                        gt_data[:,:, [0,1]] = gt_data[:,:,[1,0]]
                        pred_data[:,:,[0,1]] = pred_data[:,:,[1,0]]

                    multi_sample_pred.append(pred_data)

                all_curv = []
                for a in range(gt_data.shape[0]):
                    gt_xy = gt_data[a,:,:2]
                    curv = []
                    for i in range(1,19):
                        num = 2 * np.linalg.norm(gt_xy[i+1] - gt_xy[i]) * np.linalg.norm(gt_xy[i-1] - gt_xy[i])
                        den = np.linalg.norm(gt_xy[i+1] - gt_xy[i])  * np.linalg.norm(gt_xy[i] - gt_xy[i-1]) * np.linalg.norm(gt_xy[i+1] - gt_xy[i-1])
                        curv.append(num/den)
                    all_curv.append(curv)
                all_curv = np.round(np.array(all_curv),4)

                all_pred_curv = []
                preds = np.stack(multi_sample_pred) #(#sampling, #agent, seq len, # axis)
                for a in range(gt_data.shape[0]):
                    gt_xy = preds[:,a,:,:2].mean(axis=0)
                    curv = []
                    for i in range(1, 19):
                        num = 2 * np.linalg.norm(gt_xy[i + 1] - gt_xy[i]) * np.linalg.norm(gt_xy[i - 1] - gt_xy[i])
                        den = np.linalg.norm(gt_xy[i + 1] - gt_xy[i]) * np.linalg.norm(
                            gt_xy[i] - gt_xy[i - 1]) * np.linalg.norm(gt_xy[i + 1] - gt_xy[i - 1])
                        curv.append(num / den)
                    all_pred_curv.append(curv)
                all_pred_curv = np.round(np.array(all_pred_curv), 4)

                n_agent = gt_data.shape[0]
                n_frame = gt_data.shape[1]

                fig, ax = plt.subplots()
                title = ",".join([str(int(elt)) for elt in frame_numbers[:8]]) + ' -->\n'
                title += ",".join([str(int(elt)) for elt in frame_numbers[8:]])
                ax.set_title(title, fontsize=9)
                fig.tight_layout()


                ln_gt = []
                all_ln_pred = []


                for i in range(n_agent):
                    ln_gt.append(ax.plot([], [], colors[i] + '--')[0])
                    ln_pred = []
                    for _ in range(20):
                        ln_pred.append(ax.plot([], [], colors[i], alpha=0.3, linewidth=1)[0])
                    all_ln_pred.append(ln_pred)


                def init():
                    ax.imshow(frame)

                def update_dot(num_t):
                    print(num_t)
                    cap.set(1, frame_numbers[num_t])
                    _, frame = cap.read()
                    ax.imshow(frame)

                    for i in range(n_agent):
                        ln_gt[i].set_data(gt_data[i, :num_t, 0], gt_data[i, :num_t, 1])

                        for j in range(20):
                            all_ln_pred[i][j].set_data(multi_sample_pred[j][i, :num_t, 0], multi_sample_pred[j][i, :num_t, 1])


                ani = FuncAnimation(fig, update_dot, frames=n_frame, interval=1, init_func=init())

                # writer = PillowWriter(fps=3000)
                ani.save("D:\crowd\\fig\eth/eeth_f" + str(int(frame_numbers[0])) + "_agent" + str(agent_rng[0]) +"to" +str(agent_rng[-1]) +".gif", fps=4)




    def draw_traj(self, data_loader, num_samples):

        import cv2
        # read video
        cap = cv2.VideoCapture('D:\crowd\ewap_dataset\seq_eth\seq_eth.avi')
        frame_number=880
        cap.set(1, frame_number)
        ret, frame = cap.read()

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.imshow(frame)

        h = np.loadtxt('D:\crowd\ewap_dataset\seq_eth\H.txt')
        inv_h_t = np.linalg.pinv(np.transpose(h))

        total_traj = 0
        with torch.no_grad():
            for batch in data_loader:
                (obs_traj, fut_traj, obs_traj_rel, fut_traj_rel, non_linear_ped,
                 loss_mask, seq_start_end, obs_frames, pred_frames) = batch
                total_traj += fut_traj.size(1)

                (dist_fc_inputMx, muSharedMx, stdSharedMx) \
                    = self.encoderMx(obs_traj_rel, seq_start_end)

                all_pred = []
                for _ in range(num_samples):
                    zSharedMx = sample_gaussian(self.device, muSharedMx, stdSharedMx)
                    decoder_h = torch.cat([dist_fc_inputMx, zSharedMx], dim=1).unsqueeze(0)
                    decoder_c = torch.zeros(self.num_layers, obs_traj.size(1), self.decoder_h_dim).to(self.device)

                    pred_fut_traj_rel = self.decoderMy(
                        obs_traj[-1],
                        obs_traj_rel[-1],
                        (decoder_h, decoder_c),
                        seq_start_end,
                    )
                    pred_fut_traj = relative_to_abs(
                        pred_fut_traj_rel, obs_traj[-1]
                    )
                    all_pred.append(pred_fut_traj)
                all_pred = torch.stack(all_pred)

                colors = np.array(
                    ['red', 'magenta', 'lightgreen', 'slateblue', 'blue', 'darkgreen', 'darkorange',
                     'gray', 'purple', 'turquoise', 'midnightblue', 'olive', 'black', 'pink','burlywood',  'yellow'])


                # all 20 traj for each agent
                rng = range(148, 151)
                # rng = range(seq_start_end[17][0], seq_start_end[17][1])
                rng = range(504, 511)

                num_ped=len(rng)
                # sub = [221, 222, 223, 224]
                sub = []
                for i in range(9):
                    sub.append(int('33'+str(i+1)))

                #####################################################################
                ## on the  frame




                rng = range(12,15)

                rng = range(3,5)
                rng = range(0,3)
                rng = range(7,9)

                rng = range(12,15)

                for idx in range(len(rng)):
                    one_ped = rng[idx]
                    if idx ==0:
                        frame_number = obs_frames[one_ped, -1]
                        cap.set(1, frame_number)
                        ret, frame = cap.read()

                        fig = plt.figure(figsize=(8, 8))
                        ax = fig.add_subplot(111)
                        ax.imshow(frame)

                    obs_real = obs_traj[:, one_ped]
                    obs_real = np.concatenate([obs_real, np.ones((self.obs_len, 1))], axis=1)
                    obs_pixel = np.matmul(obs_real, inv_h_t)
                    obs_pixel /= np.expand_dims(obs_pixel[:, 2], 1)
                    # obs_real_back = np.matmul(obs_pixel, np.transpose(h))
                    # obs_real_back /= np.expand_dims(obs_real_back[:, 2], 1)
                    # ax.scatter(obs_pixel[:, 1], obs_pixel[:, 0], color=colors[0], alpha=0.5, marker='$||$', s=25)
                    plt.plot(obs_pixel[:, 1], obs_pixel[:, 0], '-', color=colors[0], alpha=0.5)

                    gt_real = fut_traj[:, one_ped]
                    gt_real = np.concatenate([gt_real, np.ones((self.pred_len, 1))], axis=1)
                    gt_pixel = np.matmul(gt_real, inv_h_t)
                    gt_pixel /= np.expand_dims(gt_pixel[:, 2], 1)
                    # obs_real_back = np.matmul(obs_pixel, np.transpose(h))
                    # obs_real_back /= np.expand_dims(obs_real_back[:, 2], 1)
                    # ax.scatter(gt_pixel[:, 1], gt_pixel[:, 0], color=colors[0], alpha=0.5, marker='|', s=25)
                    plt.plot(gt_pixel[:, 1], gt_pixel[:, 0], '--', color=colors[0], alpha=0.5)


                    pred_real = all_pred[0, :, one_ped].numpy()
                    pred_pixel = np.concatenate([pred_real, np.ones((self.pred_len, 1))], axis=1)
                    pred_pixel = np.matmul(pred_pixel, inv_h_t)
                    pred_pixel /= np.expand_dims(pred_pixel[:, 2], 1)
                    # pred_real_back = np.matmul(pred_pixel, np.transpose(h))
                    # pred_real_back /= np.expand_dims(pred_real_back[:, 2], 1)
                    # ax.scatter(pred_pixel[:, 1], pred_pixel[:, 0], color=colors[2], alpha=1, marker='.', s=13)
                    plt.plot(pred_pixel[:, 1], pred_pixel[:, 0], '--', color=colors[2], alpha=1)

                # ani = animation.FuncAnimation(fig, update_dot, frames=gen_dot, interval=5, init_func=init)

                #####################################################################
                # pred_traj collision rate

                num_samples_idx = 2
                threshold= 0.1
                coll_info = {}
                for p in range(self.pred_len):
                    pred = all_pred[num_samples_idx, p, rng[0]:rng[0] + num_ped]
                    curr1 = pred.repeat(num_ped, 1)
                    curr2 = self.repeat(pred, num_ped)
                    dist = torch.sqrt(torch.pow(curr1 - curr2, 2).sum(1)).numpy()

                    dist = dist.reshape(num_ped, num_ped)
                    diff_agent_idx = np.triu_indices(num_ped, k=1)
                    diff_agent_dist = dist[diff_agent_idx]
                    coll_idx = np.where(diff_agent_dist < threshold)[0]
                    coll_agents = []
                    coll_dist=[]
                    for idx in coll_idx:
                        coll_agents.append([diff_agent_idx[0][idx], diff_agent_idx[1][idx]])
                        coll_dist.append(dist[diff_agent_idx[0][idx]][diff_agent_idx[1][idx]])
                    coll_info.update({p:[coll_agents, coll_dist]})
                print(coll_info)
                #####################################################################
                # 20 traj for each agent
                fig = plt.figure(figsize=(12, 8))

                fig.tight_layout()
                s = -1
                for k in rng:
                    s +=1
                    if s ==9:
                        break
                    obs = obs_traj[:, k]
                    pred = all_pred[:, :, k]
                    gt = fut_traj[:, k]
                    ax = fig.add_subplot(sub[s])

                    for i in range(20):
                        if i <10:
                            marker = '+'
                            size = 15
                        else:
                            marker='^'
                            size=10
                        ax.scatter(pred[i, :, 0], pred[i, :, 1], color=colors[i % 10], alpha=0.5, marker=marker, s=size)
                    # ax.scatter(obs[0, 0], obs[0, 1], color='r', marker='^', s=10, alpha=0.5)
                    ax.scatter(obs[:,0], obs[:,1], color='black', alpha=1, marker='o', s=7)
                    ax.scatter(gt[:,0], gt[:,1], color='black', alpha=1, marker='*', s=10)
                # plt.legend(loc=3, shadow=False, scatterpoints=1, prop={'size': 8})
                plt.show()

                #####################################################################
                # all agents with one traj
                fig = plt.figure(figsize=(11, 7))
                # fig = plt.figure(figsize=(5, 3))
                ax = fig.add_subplot(111)
                fig.tight_layout()
                a = -1
                for k in rng:
                    a += 1
                    obs = obs_traj[:, k]
                    gt = fut_traj[:, k]
                    if a ==0:
                        ax.scatter(obs[:, 0], obs[:, 1], color=colors[a], alpha=0.5, marker='o', s=7,
                                   label='past')
                        ax.scatter(gt[:, 0], gt[:, 1], color=colors[a], alpha=0.5, marker='*', s=10,
                                   label='gt_future')
                    else:
                        ax.scatter(obs[:, 0], obs[:, 1], color=colors[a], alpha=0.5, marker='o', s=7)
                        ax.scatter(gt[:, 0], gt[:, 1], color=colors[a], alpha=0.5, marker='*', s=10)

                for i in range(self.pred_len):
                    pred = all_pred[num_samples_idx, i]
                    a = -1
                    for k in rng:
                        a +=1
                        if i ==0 and a==0:
                            ax.scatter(pred[k,0], pred[k,1],
                                       color=colors[a], alpha=1, marker='^', s=13, label='pred_future')
                        else:
                            ax.scatter(pred[k,0], pred[k,1],
                                       color=colors[a], alpha=1, marker='^', s=13)

                    pred = all_pred[num_samples_idx, i, rng]
                    for c in range(len(coll_info[i][0])):
                        agent_pair =  coll_info[i][0][c]
                        if coll_info[i][1][c] < 0.1:
                            ax.scatter(pred[agent_pair[0],0], pred[agent_pair[0],1],
                                       color=colors[i], alpha=0.2, marker='D', s=150)
                            ax.scatter(pred[agent_pair[1],0], pred[agent_pair[1],1],
                                       color=colors[i], alpha=0.2, marker='D', s=150)
                        ax.text(pred[agent_pair[1],0], pred[agent_pair[1],1], str(np.round(coll_info[i][1][c],2)), fontsize=8)

                plt.legend(loc=0, shadow=False, scatterpoints=1, prop={'size': 8})
                plt.show()


                #####################################################################
                # batch observed dist.
                diff_agent_dist = []
                for s_e in seq_start_end:
                    start = s_e[0].item()
                    end = s_e[1].item()
                    num_ped = end- start
                    for p in range(self.obs_len):
                        obs = obs_traj[p, start:end]
                        curr1 = obs.repeat(num_ped, 1)
                        curr2 = self.repeat(obs, num_ped)
                        dist = torch.sqrt(torch.pow(curr1 - curr2, 2).sum(1)).numpy()

                        dist = dist.reshape(num_ped, num_ped)
                        diff_agent_idx = np.triu_indices(num_ped, k=1)
                        diff_agent_dist.extend(list(dist[diff_agent_idx]))

                np.array(diff_agent_dist).min()



                # arr_img = plt.imread(os.path.join(dataset_root, 'images', path), format='jpg')
                # ax.annotate(str(cls), xy=(x0, y0), xycoords="data", color=color, fontsize=8,
                #             bbox=dict(boxstyle='round,pad=0.2', fc=bc, alpha=0.3))

                # ax.axes.get_xaxis().set_visible(False)
                # ax.axes.get_yaxis().set_visible(False)
                # plt.legend(loc=4, shadow=False, scatterpoints=1, prop={'size': 8})

                # fig.subplots_adjust(wspace=0.03)
                # fig.suptitle('lambda=' + str(lamb))
                # x_axis = list(range(0, n_data, 2))
                # x_axis = list(range(0, n_data, int(n_data/2)))
                # x_axis.append(n_data)

                # ax.set_xticklabels([])
                # ax.set_xticklabels(['(0,1)', '(0.5, 0.5)', '(1,0)'])
                # ax.set_yticklabels([])
                # ax.set_xticks(['(0,1)', '(0.5, 0.5)', '(1,0)'])
                # ax.set_xticks(x_axis)
                # ax.xaxis.set_ticklabels(np.arange(0, 1.1, 0.5))
                # ax.set_title('alpha')

    def check_dist_stat(self, data_loader):

        total_traj = 0
        diff_agent_dist = []
        with torch.no_grad():
            for batch in data_loader:
                (obs_traj, fut_traj, obs_traj_rel, fut_traj_rel, non_linear_ped,
                 loss_mask, seq_start_end) = batch
                total_traj += fut_traj.size(1)
                #####################################################################
                # batch observed dist.

                for s_e in seq_start_end:
                    start = s_e[0].item()
                    end = s_e[1].item()
                    num_ped = end - start
                    for p in range(self.obs_len):
                        obs = obs_traj[p, start:end]
                        curr1 = obs.repeat(num_ped, 1)
                        curr2 = self.repeat(obs, num_ped)
                        dist = torch.sqrt(torch.pow(curr1 - curr2, 2).sum(1)).numpy()

                        dist = dist.reshape(num_ped, num_ped)
                        diff_agent_idx = np.triu_indices(num_ped, k=1)
                        diff_agent_dist.extend(list(dist[diff_agent_idx]))
                        idx = np.where(dist[diff_agent_idx] < 0.2)
                        if len(idx[0]) > 0:
                            print(dist[diff_agent_idx][idx])

        print('min: ', np.round(np.array(diff_agent_dist).min(),4))
        print('max: ', np.round(np.array(diff_agent_dist).max(),4))
        print('avg: ', np.round(np.array(diff_agent_dist).mean(),4))
        print('std: ', np.round(np.array(diff_agent_dist).std(),4))
                #####################################################################


    ####
    def viz_init(self):
        self.viz.close(env=self.name + '/lines', win=self.win_id['loss_recon'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['loss_kl'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['total_loss'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['test_loss_recon'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['test_loss_kl'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['test_total_loss'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['ade_min'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['fde_min'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['ade_avg'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['fde_avg'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['ade_std'])
        self.viz.close(env=self.name + '/lines', win=self.win_id['fde_std'])

    ####
    def visualize_line(self):

        # prepare data to plot
        data = self.line_gather.data
        iters = torch.Tensor(data['iter'])
        loss_recon = torch.Tensor(data['loss_recon'])
        loss_kl = torch.Tensor(data['loss_kl'])
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
            X=iters, Y=total_loss, env=self.name + '/lines',
            win=self.win_id['total_loss'], update='append',
            opts=dict(xlabel='iter', ylabel='vae loss',
                      title='VAE loss'),
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
            opts=dict(xlabel='iter', ylabel='vae loss',
                      title='Test VAE loss'),
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
        else:
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


        mkdirs(self.ckpt_dir)

        torch.save(self.encoderMx, encoderMx_path)
        torch.save(self.encoderMy, encoderMy_path)
        torch.save(self.decoderMy, decoderMy_path)
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

        if self.device == 'cuda':
            self.encoderMx = torch.load(encoderMx_path)
            self.encoderMy = torch.load(encoderMy_path)
            self.decoderMy = torch.load(decoderMy_path)
        else:
            self.encoderMx = torch.load(encoderMx_path, map_location='cpu')
            self.encoderMy = torch.load(encoderMy_path, map_location='cpu')
            self.decoderMy = torch.load(decoderMy_path, map_location='cpu')
