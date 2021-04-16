import os

import torch.optim as optim
# -----------------------------------------------------------------------------#
from utils import DataGather, mkdirs, grid2gif2, apply_poe, sample_gaussian, sample_gumbel_softmax
from model import *
from loss import kl_two_gaussian, displacement_error, final_displacement_error
from utils_sgan import relative_to_abs, integrate_samples
from data.loader import data_loader
import imageio

import matplotlib.pyplot as plt
from torch.distributions import RelaxedOneHotCategorical as concrete
from torch.distributions import OneHotCategorical as discrete
from torch.distributions import kl_divergence
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from gmm2d import GMM2D

###############################################################################

class Solver(object):

    ####
    def __init__(self, args):

        self.args = args

        # self.name = '%s_pred_len_%s_zS_%s_embedding_dim_%s_enc_h_dim_%s_dec_h_dim_%s_mlp_dim_%s_pool_dim_%s_lr_%s_klw_%s' % \
        #             (args.dataset_name, args.pred_len, args.zS_dim, 16, args.encoder_h_dim, args.decoder_h_dim, args.mlp_dim, args.pool_dim, args.lr_VAE, args.kl_weight)
        # self.name = '%s_pred_len_%s_zS_%s_dr_mlp_%s_dr_rnn_%s_enc_h_dim_%s_dec_h_dim_%s_mlp_dim_%s_pool_dim_%s_lr_%s_klw_%s' % \
        #             (args.dataset_name, args.pred_len, args.zS_dim, args.dropout_mlp, args.dropout_rnn, args.encoder_h_dim,
        #              args.decoder_h_dim, args.mlp_dim, args.pool_dim, args.lr_VAE, args.kl_weight)
        self.name = '%s_pred_len_%s_zS_%s_dr_mlp_%s_dr_rnn_%s_enc_h_dim_%s_dec_h_dim_%s_mlp_dim_%s_attn_%s_lr_%s_klw_%s' % \
                    (args.dataset_name, args.pred_len, args.zS_dim, args.dropout_mlp, args.dropout_rnn, args.encoder_h_dim,
                     args.decoder_h_dim, args.mlp_dim, '1', args.lr_VAE, args.kl_weight)

        # to be appended by run_id

        # self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = args.device
        self.temp=1.99
        self.dt=0.4
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

        if self.ckpt_load_iter == 0 or args.dataset_name =='all':  # create a new model
            self.encoderMx = Encoder(
                args.zS_dim,
                enc_h_dim=args.encoder_h_dim,
                mlp_dim=args.mlp_dim,
                attention=args.attention,
                batch_norm=args.batch_norm,
                num_layers=args.num_layers,
                dropout_mlp=args.dropout_mlp,
                dropout_rnn=args.dropout_rnn).to(self.device)
            self.encoderMy = EncoderY(
                args.zS_dim,
                enc_h_dim=args.encoder_h_dim,
                mlp_dim=args.mlp_dim,
                attention=args.attention,
                batch_norm=args.batch_norm,
                num_layers=args.num_layers,
                dropout_mlp=args.dropout_mlp,
                dropout_rnn=args.dropout_rnn,
                device=self.device).to(self.device)
            self.decoderMy = Decoder(
                args.pred_len,
                dec_h_dim=self.decoder_h_dim,
                enc_h_dim=args.encoder_h_dim,
                z_dim=args.zS_dim,
                mlp_dim=args.mlp_dim,
                num_layers=args.num_layers,
                device=args.device,
                dropout_mlp=args.dropout_mlp,
                dropout_rnn=args.dropout_rnn,
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

        if self.ckpt_load_iter != self.max_iter:
            print("Initializing train dataset")
            _, self.train_loader = data_loader(self.args, train_path)
            print("Initializing val dataset")
            _, self.val_loader = data_loader(self.args, val_path)

            print(
                'There are {} iterations per epoch'.format(len(self.train_loader.dataset) / args.batch_size)
            )
        print('...done')

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


            (encX_h_feat, logitX) \
                = self.encoderMx(obs_traj, seq_start_end, train=True)
            (encY_h_feat, logitY) \
                = self.encoderMy(obs_traj[-1], fut_traj_rel, seq_start_end, encX_h_feat, train=True)

            p_dist = discrete(logits=logitX)
            q_dist = discrete(logits=logitY)
            relaxed_q_dist = concrete(logits=logitY, temperature=self.temp)


            # 첫번째 iteration 디코더 인풋 = (obs_traj_rel의 마지막 값, (hidden_state, cell_state))
            # where hidden_state = "인코더의 마지막 hidden_layer아웃풋과 그것으로 만든 max_pooled값을 concat해서 mlp 통과시켜만든 feature인 noise_input에다 noise까지 추가한값)"
            fut_rel_pos_dist = self.decoderMy(
                obs_traj[-1],
                encX_h_feat,
                relaxed_q_dist.rsample(),
                fut_traj
            )


            ################# validate integration #################
            # a = integrate_samples(fut_traj_rel, obs_traj[-1][:, :2])
            # d = a - fut_traj[:, :, :2]
            # b = relative_to_abs(fut_traj_rel, obs_traj[-1][:, :2])
            # e = b - fut_traj[:, :, :2]
            # d==e
            ####################################################################

            ################## total loss for vae ####################
            # loglikelihood = fut_rel_pos_dist.log_prob(torch.reshape(fut_traj_rel, [batch, self.pred_len, 2])).sum().div(batch)

            # log_p_yt_xz=torch.clamp(fut_rel_pos_dist.log_prob(torch.reshape(fut_traj_rel, [batch, self.pred_len, 2])), max=6)
            # print(">>>max:", log_p_yt_xz.max(), log_p_yt_xz.min(), log_p_yt_xz.mean())
            # loglikelihood = log_p_yt_xz.sum().div(batch)
            loglikelihood = fut_rel_pos_dist.log_prob(fut_traj_rel).sum().div(batch)

            loss_kl = kl_divergence(q_dist, p_dist).sum().div(batch)
            loss_kl = torch.clamp(loss_kl, min=0.07)
            # print('log_likelihood:', loglikelihood.item(), ' kl:', loss_kl.item())

            elbo = loglikelihood - self.kl_weight * loss_kl
            vae_loss = -elbo


            self.optim_vae.zero_grad()
            vae_loss.backward()
            self.optim_vae.step()


            # save model parameters
            if iteration % self.ckpt_save_iter == 0:
                self.save_checkpoint(iteration)


            # (visdom) insert current line stats
            if self.viz_on and (iteration % self.viz_ll_iter == 0):
                ade_min, fde_min, \
                ade_avg, fde_avg, \
                ade_std, fde_std, \
                test_loss_recon, test_loss_kl, test_vae_loss = self.evaluate_dist(self.val_loader, 20, loss=True)
                self.line_gather.insert(iter=iteration,
                                        loss_recon=-loglikelihood.item(),
                                        loss_kl=loss_kl.item(),
                                        total_loss=vae_loss.item(),
                                        ade_min=ade_min,
                                        fde_min=fde_min,
                                        ade_avg=ade_avg,
                                        fde_avg=fde_avg,
                                        ade_std=ade_std,
                                        fde_std=fde_std,
                                        test_loss_recon=-test_loss_recon.item(),
                                        test_loss_kl=test_loss_kl.item(),
                                        test_total_loss=test_vae_loss.item(),
                                        )
                prn_str = ('[iter_%d (epoch_%d)] vae_loss: %.3f ' + \
                              '(recon: %.3f, kl: %.3f)\n' + \
                              'ADE min: %.2f, FDE min: %.2f, ADE avg: %.2f, FDE avg: %.2f\n'
                          ) % \
                          (iteration, epoch,
                           vae_loss.item(), -loglikelihood.item(), loss_kl.item(),
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

        loss_recon = loss_kl = vae_loss = 0

        all_ade =[]
        all_fde =[]
        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_vel, fut_traj_vel, seq_start_end, obs_frames, fut_frames, past_obst,
                 fut_obst) = batch
                batch_size = obs_traj_vel.size(1)
                total_traj += fut_traj.size(1)

                (encX_h_feat, logitX) \
                    = self.encoderMx(obs_traj, seq_start_end)
                p_dist = discrete(logits=logitX)
                relaxed_p_dist = concrete(logits=logitX, temperature=self.temp)

                if loss:
                    (encY_h_feat, logitY) \
                        = self.encoderMy(obs_traj[-1], fut_traj_vel, seq_start_end, encX_h_feat)

                    q_dist = discrete(logits=logitY)
                    fut_rel_pos_dist = self.decoderMy(
                        obs_traj[-1],
                        encX_h_feat,
                        relaxed_p_dist.rsample()
                    )
                    # fut_rel_pos_dist = self.decoderMy(
                    #     obs_traj[-1],
                    #     encX_h_feat,
                    #     relaxed_p_dist.rsample((num_samples,)),
                    #     num_samples=num_samples
                    # )

                    ################## total loss for vae ####################
                    loglikelihood = fut_rel_pos_dist.log_prob(fut_traj_vel).sum().div(batch_size)

                    kld = kl_divergence(q_dist, p_dist).sum().div(batch_size)
                    kld = torch.clamp(kld, min=0.07)
                    elbo = loglikelihood - self.kl_weight * kld
                    vae_loss -=elbo
                    loss_recon +=loglikelihood
                    loss_kl +=kld

                ade, fde = [], []
                for _ in range(num_samples):
                    fut_rel_pos_dist = self.decoderMy(
                        obs_traj[-1],
                        encX_h_feat,
                        relaxed_p_dist.rsample()
                    )
                    pred_fut_traj_rel = fut_rel_pos_dist.rsample()

                    pred_fut_traj=integrate_samples(pred_fut_traj_rel, obs_traj[-1][:, :2], dt=self.dt)


                    ade.append(displacement_error(
                        pred_fut_traj, fut_traj[:,:,:2], mode='raw'
                    ))
                    fde.append(final_displacement_error(
                        pred_fut_traj[-1], fut_traj[-1,:,:2], mode='raw'
                    ))
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
                   loss_recon/b, loss_kl/b, vae_loss/b
        else:
            return ade_min, fde_min, \
                   ade_avg, fde_avg, \
                   ade_std, fde_std





    def evaluate_dtw(self, data_loader, num_samples):
        self.set_mode(train=False)
        all_dist = []
        from dtaidistance import dtw_ndim
        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_vel, fut_traj_vel, seq_start_end, obs_frames, fut_frames, past_obst,
                 fut_obst) = batch
                batch_size = fut_traj.size(1)

                (encX_h_feat, logitX) \
                    = self.encoderMx(obs_traj, seq_start_end)
                relaxed_p_dist = concrete(logits=logitX, temperature=self.temp)

                dist_20samples = [] # (20, # seq, 12)
                for _ in range(num_samples):
                    fut_rel_pos_dist = self.decoderMy(
                        obs_traj[-1],
                        encX_h_feat,
                        relaxed_p_dist.rsample()
                    )
                    pred_fut_traj_rel = fut_rel_pos_dist.rsample()

                    pred_fut_traj=integrate_samples(pred_fut_traj_rel, obs_traj[-1][:, :2], dt=self.dt)

                    dtw_dist = []
                    for i in range(batch_size):
                        dtw_dist.append(dtw_ndim.distance(pred_fut_traj[:,i].cpu().numpy(), fut_traj[:, i,:2].cpu().numpy()))
                    dist_20samples.append(dtw_dist)
                all_dist.append(np.array(dist_20samples))

            all_dist=np.concatenate(all_dist, axis=1) #(20,70,12)
            print('all_coll: ', all_dist.shape)
            dtw_min=all_dist.min(axis=0).mean()
            dtw_avg=all_dist.mean(axis=0).mean()
            dtw_std=all_dist.std(axis=0).mean()
        self.set_mode(train=True)
        return dtw_min, dtw_avg, dtw_std



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
                                             binary_dilation(obs_map, iterations=1),
                                             kx=1, ky=1)

        old_shape = predicted_trajs.shape
        predicted_trajs = predicted_trajs.reshape((-1,2))

        # plt.imshow(obs_map)
        # import cv2
        # cap = cv2.VideoCapture(os.path.join('D:\crowd\datasets/nmap\map2', self.dataset_name + '_video.avi'))
        # cap.set(1, int(880))
        # _, ff = cap.read()
        # plt.imshow(ff)
        # for i in range(12):
        #     plt.scatter(predicted_trajs[i,0], predicted_trajs[i,1], s=1)
        #
        # a = binary_dilation(obs_map, iterations=1)
        # plt.imshow(a)
        # for i in range(12):
        #     plt.scatter(predicted_trajs[i,0], predicted_trajs[i,1], s=1)

        traj_obs_values = interp_obs_map(predicted_trajs[:, 1], predicted_trajs[:, 0], grid=False)
        traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1]))
        num_viol_trajs = np.sum(traj_obs_values.max(axis=1) > 0,
                                dtype=float)  # 20개 case 각각에 대해 12 future time중 한번이라도(12개중 max) 충돌이 있었나 확인

        return num_viol_trajs, traj_obs_values.sum(axis=1)

    def map_collision(self, data_loader, num_samples=20):

        obs_map = imageio.imread(os.path.join('../datasets/nmap/map', self.dataset_name + '_map.png'))
        h = np.loadtxt(os.path.join('../datasets/nmap/map', self.dataset_name +'_H.txt'))
        inv_h_t = np.linalg.pinv(np.transpose(h))

        total_traj = 0
        total_viol = 0
        min_viol = []
        avg_viol = []
        std_viol = []
        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_vel, fut_traj_vel, seq_start_end, obs_frames, fut_frames, past_obst,
                 fut_obst) = batch
                total_traj += fut_traj.size(1)

                (encX_h_feat, logitX) \
                    = self.encoderMx(obs_traj, seq_start_end, train=False)
                relaxed_p_dist = concrete(logits=logitX, temperature=self.temp)




                for s, e in seq_start_end:
                    agent_rng = range(s, e)

                    multi_sample_pred = []
                    for _ in range(num_samples):
                        fut_rel_pos_dist = self.decoderMy(
                            obs_traj[-1],
                            encX_h_feat,
                            relaxed_p_dist.rsample()
                        )

                        pred_fut_traj_vel = fut_rel_pos_dist.rsample()
                        pred_fut_traj = integrate_samples(pred_fut_traj_vel, obs_traj[-1][:, :2], dt=self.dt)

                        pred_data = []
                        for idx in range(len(agent_rng)):
                            one_ped = agent_rng[idx]
                            pred_real = pred_fut_traj[:, one_ped].detach().cpu().numpy()
                            pred_pixel = np.concatenate([pred_real, np.ones((self.pred_len, 1))], axis=1)
                            pred_pixel = np.matmul(pred_pixel, inv_h_t)
                            pred_pixel /= np.expand_dims(pred_pixel[:, 2], 1)
                            pred_data.append(pred_pixel)

                        pred_data = np.stack(pred_data)
                        pred_data[:,:,[0,1]] = pred_data[:,:,[1,0]]

                        multi_sample_pred.append(pred_data)

                    for a in range(len(agent_rng)):
                        num_viol_trajs, viol20 = self.compute_obs_violations(np.array(multi_sample_pred)[:,a,:,:2], obs_map)
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





    def plot_traj_var(self, data_loader, num_samples=20):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        import cv2
        gif_path = "D:\crowd\\fig\\runid" + str(self.run_id)
        mkdirs(gif_path)
        # read video
        cap = cv2.VideoCapture('D:\crowd\ewap_dataset\seq_eth\seq_eth.avi')

        colors = ['r', 'g', 'y', 'm', 'c', 'k', 'w', 'b']
        h = np.loadtxt('D:\crowd\ewap_dataset\seq_eth\H.txt')
        inv_h_t = np.linalg.pinv(np.transpose(h))

        total_traj = 0
        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_rel, fut_traj_rel, non_linear_ped,
                 loss_mask, seq_start_end, obs_frames, pred_frames) = batch
                batch_size = obs_traj_rel.size(1)
                total_traj += fut_traj.size(1)

                (encX_h_feat, logitX) \
                    = self.encoderMx(obs_traj, seq_start_end)
                relaxed_p_dist = concrete(logits=logitX, temperature=self.temp)


                agent_rng = range(224,226)
                # frame_number = obs_frames[95][-1]
                frame_numbers = np.concatenate([obs_frames[agent_rng[0]], pred_frames[agent_rng[0]]])
                frame_number = frame_numbers[0]
                cap.set(1, frame_number)
                ret, frame = cap.read()
                multi_sample_pred = []

                for _ in range(num_samples):
                    fut_rel_pos_dist = self.decoderMy(
                        obs_traj[-1],
                        encX_h_feat,
                        relaxed_p_dist.rsample()
                    )
                    pred_fut_traj_rel = fut_rel_pos_dist.rsample()
                    pred_fut_traj=integrate_samples(pred_fut_traj_rel, obs_traj[-1][:, :2], dt=self.dt)

                    gt_data, pred_data = [], []

                    for idx in range(len(agent_rng)):
                        one_ped = agent_rng[idx]
                        obs_real = obs_traj[:, one_ped,:2]
                        obs_real = np.concatenate([obs_real, np.ones((self.obs_len, 1))], axis=1)
                        obs_pixel = np.matmul(obs_real, inv_h_t)
                        obs_pixel /= np.expand_dims(obs_pixel[:, 2], 1)

                        gt_real = fut_traj[:, one_ped, :2]
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

                ani.save(gif_path + "/eth_f" + str(int(frame_numbers[0])) + "_agent" + str(agent_rng[0]) +"to" +str(agent_rng[-1]) +".gif", fps=4)




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
