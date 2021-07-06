import os

import torch.optim as optim
# -----------------------------------------------------------------------------#
from utils import DataGather, mkdirs, grid2gif2, apply_poe, sample_gaussian, sample_gumbel_softmax
from model import *
from loss import kl_two_gaussian, displacement_error, final_displacement_error
from utils_sgan import relative_to_abs, integrate_samples
from data.loader import data_loader
import imageio
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
from torch.distributions import RelaxedOneHotCategorical as concrete
from torch.distributions import OneHotCategorical as discrete
from torch.distributions import kl_divergence
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from transformer.model import EncoderX, EncoderY, DecoderY
from transformer.functional import subsequent_mask

###############################################################################

class Solver(object):

    ####
    def __init__(self, args):

        self.args = args

        self.name = '%s_pred_len_%s_d_latent_%s_d_model_%s_d_ff_%s_n_layers_%s_n_head_%s_dropout_%s_klw_%s' % \
                    (args.dataset_name, args.pred_len, args.latent_dim, args.emb_size,
                     args.d_ff, args.layers, args.heads, args.dropout, args.kl_weight)


        self.device = args.device
        self.temp=1.99
        self.dt=0.4
        self.kl_weight=args.kl_weight
        self.emb_size=args.emb_size

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
        self.d_latent = args.latent_dim
        self.d_latent = args.latent_dim
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

        if self.ckpt_load_iter == 0 or args.dataset_name =='all':  # create a new model
            self.encoderX = EncoderX(enc_inp_size=2,
                                     d_latent=args.latent_dim,
                                     N = args.layers,
                                     d_model = args.emb_size,
                                     d_ff = args.d_ff,
                                     h = args.heads,
                                     dropout = args.dropout).to(self.device)
            self.encoderY = EncoderY(enc_inp_size=2,
                                     d_latent=args.latent_dim,
                                     N = args.layers,
                                     d_model = args.emb_size,
                                     d_ff = args.d_ff,
                                     h = args.heads,
                                     dropout = args.dropout).to(self.device)
            self.decoderY = DecoderY(dec_inp_size=3,
                                     dec_out_size=2,
                                     d_latent=args.latent_dim,
                                     N = args.layers,
                                     d_model = args.emb_size,
                                     d_ff = args.d_ff,
                                     h = args.heads,
                                     dropout = args.dropout).to(self.device)


        else:  # load a previously saved model
            print('Loading saved models (iter: %d)...' % self.ckpt_load_iter)
            self.load_checkpoint()
            print('...done')


        # get VAE parameters
        vae_params = \
            list(self.encoderX.parameters()) + \
            list(self.encoderY.parameters()) + \
            list(self.decoderY.parameters())

        # create optimizers
        self.optim_vae = optim.Adam(
            vae_params,
            lr=self.lr_VAE,
            betas=[self.beta1_VAE, self.beta2_VAE]
        )

        # prepare dataloader (iterable)
        print('Start loading data...')
        train_path = os.path.join(self.dataset_dir, self.dataset_name, 'test')
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
            (obs_traj, fut_traj, obs_traj_rel, fut_traj_rel, seq_start_end, obs_frames, pred_frames) = next(iterator)
            batch_size = obs_traj_rel.size(0) #=sum(seq_start_end[:,1] - seq_start_end[:,0])

            # prior_token = Variable(torch.zeros(batch_size, 1, self.emb_size))
            # posterior_token = Variable(torch.zeros(batch_size, 1, self.emb_size))
            # encX_inp = torch.cat((prior_token, obs_traj_rel), dim=1)
            # encY_inp = torch.cat((posterior_token, obs_traj_rel, fut_traj_rel), dim=1)
            encX_inp = obs_traj_rel
            encY_inp = torch.cat((obs_traj_rel, fut_traj_rel), dim=1)

            start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1).to(self.device)
            dec_input = fut_traj_rel[:,:-1]
            dec_inp=torch.cat((dec_input,torch.zeros((dec_input.shape[0],dec_input.shape[1],1)).to(self.device)),-1) # 70, 12, 3( 0이 더붙음)
            dec_inp = torch.cat((start_of_seq, dec_inp), 1) # 70, 13, 2. : 13 seq중에 맨 앞에 값이 (0,0,1)이게됨. 나머지 12개는 (x,y,0)


            encX_mask = encY_mask = None
            # encX_mask = torch.ones((encX_inp.shape[0], 1, encX_inp.shape[1] + 1)).to(self.device) # bs, 1,7의 all 1
            dec_mask=subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0],1,1).to(self.device) # bs, 12,12의 T/F인데, [t,f,f,...,f]부터 [t,t,..,t]까지 12dim의 vec가 12개

            encX_feat, prior_logit = self.encoderX(encX_inp, encX_mask)
            encY_feat, posterior_logit = self.encoderY(encY_inp, encY_mask)


            p_dist = discrete(logits=prior_logit)
            q_dist = discrete(logits=posterior_logit)
            relaxed_q_dist = concrete(logits=posterior_logit, temperature=self.temp)



            mu, logvar = self.decoderY(
                encX_feat, relaxed_q_dist.rsample(), dec_inp,
                encX_mask, dec_mask, seq_start_end, obs_traj[:, :, :2]
            )
            fut_rel_pos_dist = Normal(mu, torch.sqrt(torch.exp(logvar)))

            # ade_min, fde_min, \
            # ade_avg, fde_avg, \
            # ade_std, fde_std, \
            # test_loss_recon, test_loss_kl, test_vae_loss = self.evaluate_dist(self.val_loader, 20, loss=True)

            ################# validate integration #################
            # a = integrate_samples(fut_traj_rel, obs_traj[:, -1, :2], self.dt)
            # d = a - fut_traj[:, :, :2]
            # b = relative_to_abs(fut_traj_rel, obs_traj[:, -1, :2])
            # e = b - fut_traj[:, :, :2]
            # d==e
            ####################################################################

            ################## total loss for vae ####################
            # loglikelihood = fut_rel_pos_dist.log_prob(torch.reshape(fut_traj_rel, [batch, self.pred_len, 2])).sum().div(batch)

            # log_p_yt_xz=torch.clamp(fut_rel_pos_dist.log_prob(torch.reshape(fut_traj_rel, [batch, self.pred_len, 2])), max=6)
            # print(">>>max:", log_p_yt_xz.max(), log_p_yt_xz.min(), log_p_yt_xz.mean())
            # loglikelihood = log_p_yt_xz.sum().div(batch)
            loglikelihood = fut_rel_pos_dist.log_prob(fut_traj_rel).sum().div(batch_size)

            loss_kl = kl_divergence(q_dist, p_dist).sum().div(batch_size)
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
                (obs_traj, fut_traj, obs_traj_rel, fut_traj_rel, seq_start_end, obs_frames, fut_frames) = batch

                batch_size = obs_traj_rel.size(0)  # =sum(seq_start_end[:,1] - seq_start_end[:,0])

                encX_inp = obs_traj_rel
                encX_mask = encY_mask=None
                encX_feat, prior_logit = self.encoderX(encX_inp, encX_mask)

                p_dist = discrete(logits=prior_logit)
                relaxed_p_dist = concrete(logits=prior_logit, temperature=self.temp)

                if loss:
                    encY_inp = torch.cat((obs_traj_rel, fut_traj_rel), dim=1)
                    # dec_inp = obs_traj_rel[:, -1].unsqueeze(1)

                    start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1).to(
                        self.device)
                    dec_inp = start_of_seq

                    encY_feat, posterior_logit = self.encoderY(encY_inp, encY_mask)
                    q_dist = discrete(logits=posterior_logit)

                    mus = []
                    stds = []
                    for i in range(self.pred_len):  # 12
                        dec_mask = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(self.device)
                        mu, logvar = self.decoderY(
                            encX_feat, relaxed_p_dist.rsample(), dec_inp,
                            encX_mask, dec_mask, seq_start_end, obs_traj[:, :, :2]
                        )
                        mu = mu[:, -1:, :]
                        std = torch.sqrt(torch.exp(logvar))[:, -1:, :]
                        mus.append(mu)
                        stds.append(std)
                        dec_out = Normal(mu, std).rsample()
                        dec_out = torch.cat((dec_out,
                                             torch.zeros((dec_out.shape[0], dec_out.shape[1], 1)).to(
                                                 self.device)), -1)  # 70, i, 3( 0이 더붙음)
                        dec_inp = torch.cat((dec_inp, dec_out),1)

                    mus = torch.cat(mus, dim=1)
                    stds = torch.cat(stds, dim=1)
                    fut_rel_pos_dist = Normal(mus, stds)

                    ################## total loss for vae ####################
                    loglikelihood = fut_rel_pos_dist.log_prob(fut_traj_rel).sum().div(batch_size)

                    kld = kl_divergence(q_dist, p_dist).sum().div(batch_size)
                    kld = torch.clamp(kld, min=0.07)
                    elbo = loglikelihood - self.kl_weight * kld
                    vae_loss -=elbo
                    loss_recon +=loglikelihood
                    loss_kl +=kld

                ade, fde = [], []
                for _ in range(num_samples): # different relaxed_p_dist.rsample()
                    mus = []
                    stds = []
                    # dec_inp = obs_traj_rel[:, -1].unsqueeze(1)
                    start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1).to(
                        self.device)
                    dec_inp = start_of_seq

                    for i in range(self.pred_len):  # 12
                        dec_mask = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(self.device)
                        mu, logvar = self.decoderY(
                            encX_feat, relaxed_p_dist.rsample(), dec_inp,
                            encX_mask, dec_mask, seq_start_end, obs_traj[:, :, :2]
                        )
                        mu = mu[:, -1:, :]
                        std = torch.sqrt(torch.exp(logvar))[:, -1:, :]
                        mus.append(mu)
                        stds.append(std)
                        dec_out = Normal(mu, std).rsample()
                        dec_out = torch.cat((dec_out,
                                             torch.zeros((dec_out.shape[0], dec_out.shape[1], 1)).to(
                                                 self.device)), -1)  # 70, i, 3( 0이 더붙음)
                        dec_inp = torch.cat((dec_inp, dec_out),1)

                    mus = torch.cat(mus, dim=1)
                    stds = torch.cat(stds, dim=1)
                    fut_rel_pos_dist = Normal(mus, stds)

                    pred_fut_traj_rel = fut_rel_pos_dist.rsample()
                    pred_fut_traj=integrate_samples(pred_fut_traj_rel, obs_traj[:, -1, :2], dt=self.dt)


                    ade.append(displacement_error(
                        pred_fut_traj, fut_traj[:,:,:2], mode='raw'
                    ))
                    fde.append(final_displacement_error(
                        pred_fut_traj[:, -1], fut_traj[:, -1, :2], mode='raw'
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





    def evaluate_collision(self, data_loader, num_samples, threshold):
        self.set_mode(train=False)
        all_coll = []
        all_coll_ll = []
        total_pairs = 0
        with torch.no_grad():
            b=0
            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_rel, fut_traj_rel, seq_start_end, obs_frames, fut_frames) = batch

                batch_size = obs_traj_rel.size(0)  # =sum(seq_start_end[:,1] - seq_start_end[:,0])


                encX_inp = obs_traj_rel
                encX_mask = encY_mask=None
                encX_feat, prior_logit = self.encoderX(encX_inp, encX_mask)

                relaxed_p_dist = concrete(logits=prior_logit, temperature=self.temp)

                coll_20samples = [] # (20, # seq, 12)
                coll_20samples_ll = [] # (20, # seq, 12)
                for _ in range(num_samples):  # different relaxed_p_dist.rsample()
                    mus = []
                    stds = []
                    start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1).to(
                        self.device)
                    dec_inp = start_of_seq

                    for i in range(self.pred_len):  # 12
                        dec_mask = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(self.device)
                        mu, logvar = self.decoderY(
                            encX_feat, relaxed_p_dist.rsample(), dec_inp,
                            encX_mask, dec_mask, seq_start_end, obs_traj[:, :, :2]
                        )
                        mu = mu[:, -1:, :]
                        std = torch.sqrt(torch.exp(logvar))[:, -1:, :]
                        mus.append(mu)
                        stds.append(std)
                        dec_out = Normal(mu, std).rsample()
                        dec_out = torch.cat((dec_out,
                                             torch.zeros((dec_out.shape[0], dec_out.shape[1], 1)).to(
                                                 self.device)), -1)  # 70, i, 3( 0이 더붙음)
                        dec_inp = torch.cat((dec_inp, dec_out), 1)

                    mus = torch.cat(mus, dim=1)
                    stds = torch.cat(stds, dim=1)
                    fut_rel_pos_dist = Normal(mus, stds)

                    pred_fut_traj_rel = fut_rel_pos_dist.rsample()
                    pred_fut_traj = integrate_samples(pred_fut_traj_rel, obs_traj[:, -1, :2], dt=self.dt)
                    likelihood = torch.exp(fut_rel_pos_dist.log_prob(fut_traj_rel))

                    seq_coll = [] #64
                    seq_coll_ll = [] #64
                    for idx, (start, end) in enumerate(seq_start_end):

                        start = start.item()
                        end = end.item()
                        num_ped = end - start
                        if num_ped==1:
                            continue
                        one_frame_slide = pred_fut_traj[start:end] # (pred_len, num_ped, 2)
                        one_frame_likelihood = likelihood[start:end].prod(2)

                        frame_coll = [] #num_ped
                        frame_coll_ll = [] #num_ped
                        for i in range(self.pred_len):
                            ## distance
                            curr_frame = one_frame_slide[:,i] # frame of time=i #(num_ped,2)
                            curr1 = curr_frame.repeat(num_ped, 1)
                            curr2 = self.repeat(curr_frame, num_ped)
                            dist = torch.norm(curr1 - curr2, dim=1).cpu().numpy()
                            dist = dist.reshape(num_ped, num_ped) # all distance between all num_ped*num_ped
                            ## likelihood
                            ll1 = one_frame_likelihood[:,i].unsqueeze(1).repeat(num_ped, 1)
                            ll2 = self.repeat(one_frame_likelihood[:,i].unsqueeze(1), num_ped)
                            ll_mat = (ll1*ll2).reshape(num_ped, num_ped)
                            ## check if the distance < threshold for all pairs(num_ped C 2)
                            diff_agent_idx = np.triu_indices(num_ped, k=1) # only distinct distances of num_ped C 2(upper triange except for diag)
                            total_pairs +=len(diff_agent_idx[0])
                            diff_agent_dist = dist[diff_agent_idx]
                            diff_agent_ll = ll_mat[diff_agent_idx]
                            curr_coll_rate_ll = diff_agent_ll[(diff_agent_dist < threshold)].sum().item()
                            curr_coll_rate = (diff_agent_dist < threshold).sum()

                            frame_coll.append(curr_coll_rate)
                            frame_coll_ll.append(curr_coll_rate_ll) # frame 12
                        seq_coll.append(frame_coll) # all in seq_start_end
                        seq_coll_ll.append(frame_coll_ll)
                    coll_20samples.append(seq_coll) # 20 samples
                    coll_20samples_ll.append(seq_coll_ll)

                all_coll.append(np.array(coll_20samples)) # all batches
                all_coll_ll.append(np.array(coll_20samples_ll))

            all_coll=np.concatenate(all_coll, axis=1) #(20,70,12)
            all_coll_ll=np.concatenate(all_coll_ll, axis=1) #(20,70,12)
            print('all_coll: ', all_coll.shape)
            coll_rate_sum=all_coll.sum()
            coll_rate_min=all_coll.min(axis=0).sum()
            coll_rate_avg=all_coll.mean(axis=0).sum()
            coll_rate_std=all_coll.std(axis=0).mean()


        return coll_rate_sum, all_coll_ll.sum(), \
               coll_rate_min, all_coll_ll.min(axis=0).sum(), \
               coll_rate_avg, all_coll_ll.mean(axis=0).sum(), \
               coll_rate_std, all_coll_ll.std(axis=0).mean(), total_pairs



    def evaluate_collision_total(self, data_loader, num_samples, threshold):
        self.set_mode(train=False)

        total_pairs = 0
        with torch.no_grad():
            b=0
            all_pred20 = []
            all_ll20 = []
            all_seq_start_end = []

            for batch in data_loader:
                b+=1
                (obs_traj, fut_traj, obs_traj_rel, fut_traj_rel, seq_start_end, obs_frames, fut_frames) = batch

                batch_size = obs_traj_rel.size(0)  # =sum(seq_start_end[:,1] - seq_start_end[:,0])


                encX_inp = obs_traj_rel
                encX_mask = encY_mask=None
                encX_feat, prior_logit = self.encoderX(encX_inp, encX_mask)

                relaxed_p_dist = concrete(logits=prior_logit, temperature=self.temp)
                pred20 = [] # (20, # seq, 12)
                ll20 = [] # (20, # seq, 12)

                for _ in range(num_samples):  # different relaxed_p_dist.rsample()
                    mus = []
                    stds = []
                    start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1).to(
                        self.device)
                    dec_inp = start_of_seq

                    for i in range(self.pred_len):  # 12
                        dec_mask = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(self.device)
                        mu, logvar = self.decoderY(
                            encX_feat, relaxed_p_dist.rsample(), dec_inp,
                            encX_mask, dec_mask, seq_start_end, obs_traj[:, :, :2]
                        )
                        mu = mu[:, -1:, :]
                        std = torch.sqrt(torch.exp(logvar))[:, -1:, :]
                        mus.append(mu)
                        stds.append(std)
                        dec_out = Normal(mu, std).rsample()
                        dec_out = torch.cat((dec_out,
                                             torch.zeros((dec_out.shape[0], dec_out.shape[1], 1)).to(
                                                 self.device)), -1)  # 70, i, 3( 0이 더붙음)
                        dec_inp = torch.cat((dec_inp, dec_out), 1)

                    mus = torch.cat(mus, dim=1)
                    stds = torch.cat(stds, dim=1)
                    fut_rel_pos_dist = Normal(mus, stds)

                    pred_fut_traj_rel = fut_rel_pos_dist.rsample()
                    pred_fut_traj = integrate_samples(pred_fut_traj_rel, obs_traj[:, -1, :2], dt=self.dt)
                    likelihood = torch.exp(fut_rel_pos_dist.log_prob(fut_traj_rel))

                    pred20.append(pred_fut_traj) # 20 samples
                    ll20.append(likelihood)


                all_pred20.append(torch.stack(pred20)) # bs, 12, 2 의 list를 stack => 20, bs, 12, 2
                all_ll20.append(torch.stack(ll20))
                if len(all_seq_start_end) > 0:
                    seq_start_end +=all_seq_start_end[-1][-1][1]
                all_seq_start_end.append(seq_start_end)

            all_pred20 = torch.cat(all_pred20, dim=1)
            all_ll20 = torch.cat(all_ll20, dim=1)
            all_seq_start_end = torch.cat(all_seq_start_end, dim=0)

            all_coll = []
            all_coll_ll = []
            all_coll2 = []
            all_coll_ll2 = []
            all_coll3 = []
            all_coll_ll3 = []
            for idx, (start, end) in enumerate(all_seq_start_end):
                start = start.item()
                end = end.item()
                num_ped = end - start
                if num_ped == 1:
                    continue
                one_frame_slide = all_pred20[:, start:end]  # (20, num_ped, 12, 2)
                one_frame_likelihood = all_ll20[:, start:end].prod(-1)

                one_frame_slide= one_frame_slide.reshape(-1, one_frame_slide.shape[2], one_frame_slide.shape[3])
                one_frame_likelihood = one_frame_likelihood.reshape(-1, one_frame_likelihood.shape[2])

                frame_coll = []  # num_ped
                frame_coll_ll = []  # num_ped
                frame_coll2 = []  # num_ped
                frame_coll_ll2 = []  # num_ped
                frame_coll3 = []  # num_ped
                frame_coll_ll3 = []  # num_ped
                for i in range(self.pred_len):
                    ## distance
                    curr_frame = one_frame_slide[:, i]  # frame of time=i #(num_ped,2)
                    curr1 = curr_frame.repeat(num_ped * num_samples, 1)
                    curr2 = self.repeat(curr_frame, num_ped * num_samples)
                    dist = torch.norm(curr1 - curr2, dim=1).cpu().numpy()
                    dist = dist.reshape(num_ped * num_samples, num_ped * num_samples)  # all distance between all num_ped*num_ped

                    for i1 in range(num_ped * num_samples):
                        for i2 in range(i1+1, num_ped * num_samples):
                            if (i2 - i1 ) % num_ped ==0:
                                dist[i1,i2] = 9 # make the distance large for the same agent
                                total_pairs -=1

                    ## likelihood
                    ll1 = one_frame_likelihood[:, i].unsqueeze(1).repeat(num_ped * num_samples, 1)
                    ll2 = self.repeat(one_frame_likelihood[:, i].unsqueeze(1), num_ped * num_samples)
                    ll_mat = (ll1 * ll2).reshape(num_ped * num_samples, num_ped * num_samples)
                    ## check if the distance < threshold for all pairs(num_ped C 2)
                    diff_agent_idx = np.triu_indices(num_ped * num_samples, k=1)  # only distinct distances of num_ped C 2(upper triange except for diag)
                    total_pairs += len(diff_agent_idx[0])
                    diff_agent_dist = dist[diff_agent_idx]
                    diff_agent_ll = ll_mat[diff_agent_idx]
                    frame_coll.append((diff_agent_dist < threshold[0]).sum())
                    frame_coll_ll.append(diff_agent_ll[(diff_agent_dist < threshold[0])].sum().item())

                    frame_coll2.append((diff_agent_dist < threshold[1]).sum())
                    frame_coll_ll2.append(diff_agent_ll[(diff_agent_dist < threshold[1])].sum().item())

                    frame_coll3.append((diff_agent_dist < threshold[2]).sum())
                    frame_coll_ll3.append(diff_agent_ll[(diff_agent_dist < threshold[2])].sum().item())

                all_coll.append(np.array(frame_coll))
                all_coll_ll.append(np.array(frame_coll_ll))

                all_coll2.append(np.array(frame_coll2))
                all_coll_ll2.append(np.array(frame_coll_ll2))

                all_coll3.append(np.array(frame_coll3))
                all_coll_ll3.append(np.array(frame_coll_ll3))


            print('all_coll: ', np.stack(all_coll_ll).shape)

        return np.stack(all_coll).sum(), np.stack(all_coll_ll).sum(), \
               np.stack(all_coll2).sum(), np.stack(all_coll_ll2).sum(), \
               np.stack(all_coll3).sum(), np.stack(all_coll_ll3).sum(), total_pairs


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
                    = self.encoderX(obs_traj, seq_start_end, train=False)
                relaxed_p_dist = concrete(logits=logitX, temperature=self.temp)




                for s, e in seq_start_end:
                    agent_rng = range(s, e)

                    multi_sample_pred = []
                    for _ in range(num_samples):
                        fut_rel_pos_dist = self.decoderY(
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
        from matplotlib.animation import FuncAnimation
        import cv2
        gif_path = "D:\crowd\\fig\\runid" + str(self.run_id)
        mkdirs(gif_path)
        # read video
        cap = cv2.VideoCapture('D:\crowd\ewap_dataset\seq_'+self.dataset_name+'\seq_'+self.dataset_name+'.avi')

        colors = ['r', 'g', 'y', 'm', 'c', 'k', 'w', 'b']
        h = np.loadtxt('D:\crowd\ewap_dataset\seq_'+self.dataset_name+'\H.txt')
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

                # path = '../datasets\hotel\\test\\biwi_hotel.txt'
                # l=f.readlines()
                # data = read_file(path, 'tab')
                # framd_num=6980
                # np.where(obs_frames[:, 0] == framd_num)
                # d = data[1989:2000]
                # gt_real = d[..., -2:]
                # gt_real = np.concatenate([gt_real, np.ones((2000-1989, 1))], axis=1)
                # gt_pixel = np.matmul(gt_real, inv_h_t)
                # gt_pixel /= np.expand_dims(gt_pixel[:, 2], 1)
                #
                # fig, ax = plt.subplots()
                # cap.set(1, framd_num)
                # _, frame = cap.read()
                # ax.imshow(frame)
                # for i in range(len(d)):
                #     ax.text(gt_pixel[i][1], gt_pixel[i][0], str(int(d[:,1][i])), fontsize=10)


                (encX_h_feat, logitX) \
                    = self.encoderX(obs_traj, seq_start_end)
                relaxed_p_dist = concrete(logits=logitX, temperature=self.temp)

                # s=seq_start_end.numpy()
                # np.where(s[:,0]==63)

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
                            all_ln_pred[i][j].set_data(multi_sample_pred[j][i, :num_t, 0],
                                                       multi_sample_pred[j][i, :num_t, 1])

                for s, e in seq_start_end:
                    agent_rng = range(s, e)

                    frame_numbers = np.concatenate([obs_frames[agent_rng[0]], pred_frames[agent_rng[0]]])
                    frame_number = frame_numbers[0]
                    cap.set(1, frame_number)
                    ret, frame = cap.read()
                    multi_sample_pred = []

                    for _ in range(num_samples):
                        fut_rel_pos_dist = self.decoderY(
                            obs_traj[-1],
                            encX_h_feat,
                            relaxed_p_dist.rsample()
                        )
                        pred_fut_traj_rel = fut_rel_pos_dist.rsample()
                        pred_fut_traj = integrate_samples(pred_fut_traj_rel, obs_traj[-1][:, :2], dt=self.dt)

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

                        # if self.dataset_name == 'eth':
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


                    ani = FuncAnimation(fig, update_dot, frames=n_frame, interval=1, init_func=init())

                    # writer = PillowWriter(fps=3000)

                    ani.save(gif_path + "/" +self.dataset_name+ "_f" + str(int(frame_numbers[0])) + "_agent" + str(agent_rng[0]) +"to" +str(agent_rng[-1]) +".gif", fps=4)

    def plot_traj_var2(self, data_loader, num_samples=20):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import cv2
        gif_path = "D:\crowd\\fig\\runid" + str(self.run_id)
        mkdirs(gif_path)
        # read video
        # cap = cv2.VideoCapture('D:\crowd\ewap_dataset\seq_' + self.dataset_name + '\seq_' + self.dataset_name + '.avi')
        # cap = cv2.VideoCapture('D:\crowd/ucy_original\data/crowds_zara01.avi')
        cap = cv2.VideoCapture('D:\crowd/ucy_original\data/crowds_zara01.avi')

        colors = ['r', 'g', 'y', 'm', 'c', 'k', 'w', 'b']
        h = np.loadtxt('D:\crowd\ewap_dataset\seq_' + self.dataset_name + '\H.txt')
        # h = np.loadtxt('D:\crowd\datasets/nmap\map/zara01_H.txt')
        inv_h_t = np.linalg.pinv(np.transpose(h))

        total_traj = 0
        with torch.no_grad():
            b = 0
            for batch in data_loader:
                b += 1
                (obs_traj, fut_traj, obs_traj_rel, fut_traj_rel, seq_start_end, obs_frames, fut_frames, past_obst, fut_obst) = batch


                total_traj += fut_traj.size(1)


                def init():
                    ax.imshow(frame)

                def update_dot(num_t):
                    print(num_t)
                    cap.set(1, frame_numbers[num_t])
                    _, frame = cap.read()
                    ax.imshow(frame)

                    for i in range(n_agent):
                        ln_gt[i].set_data(gt_data[i, :num_t, 0], gt_data[i, :num_t, 1])

                for s, e in seq_start_end:
                    agent_rng = range(s, e)

                    frame_numbers = np.concatenate([fut_frames[agent_rng[0]]])
                    frame_number = frame_numbers[0]
                    cap.set(1, frame_number)
                    ret, frame = cap.read()
                    gt_data = []

                    for idx in range(len(agent_rng)):
                        one_ped = agent_rng[idx]

                        gt_real = fut_traj[:, one_ped, :2]
                        gt_real = np.concatenate([gt_real, np.ones((self.pred_len, 1))], axis=1)
                        gt_pixel = np.matmul(gt_real, inv_h_t)
                        gt_pixel /= np.expand_dims(gt_pixel[:, 2], 1)
                        gt_data.append(gt_pixel)  # (20, 3)

                    gt_data = np.stack(gt_data)
                    # if self.dataset_name == 'eth':
                    gt_data[:, :, [0, 1]] = gt_data[:, :, [1, 0]]

                    n_agent = gt_data.shape[0]
                    n_frame = gt_data.shape[1]

                    fig, ax = plt.subplots()
                    # title = ",".join([str(int(elt)) for elt in frame_numbers[:8]]) + ' -->\n'
                    # title += ",".join([str(int(elt)) for elt in frame_numbers[8:]])
                    # ax.set_title(title, fontsize=9)
                    ax.axis('off')
                    fig.tight_layout()

                    ln_gt = []

                    colors = ['r', 'g', 'y', 'm', 'c', 'blue']

                    ax.imshow(frame)
                    for i in range(n_agent):
                        plt.plot(gt_data[i,:,0], gt_data[i,:,1], c=colors[i])
                        plt.scatter(gt_data[i,-1,0], gt_data[i,-1,1], c=colors[i])

                    for i in range(n_agent):
                        ln_gt.append(ax.plot([], [], colors[i])[0])

                    ani = FuncAnimation(fig, update_dot, frames=n_frame, interval=1, init_func=init())

                    ani.save(gif_path + "/" + self.dataset_name + "_f" + str(
                        int(frame_numbers[0])) + "_agent" + str(agent_rng[0]) + "to" + str(
                        agent_rng[-1]) + ".gif", fps=4)

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
            self.encoderX.train()
            self.encoderY.train()
            self.decoderY.train()
        else:
            self.encoderX.eval()
            self.encoderY.eval()
            self.decoderY.eval()

    ####
    def save_checkpoint(self, iteration):

        encoderX_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderX.pt' % iteration
        )
        encoderY_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderY.pt' % iteration
        )
        decoderY_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_decoderY.pt' % iteration
        )


        mkdirs(self.ckpt_dir)

        torch.save(self.encoderX, encoderX_path)
        torch.save(self.encoderY, encoderY_path)
        torch.save(self.decoderY, decoderY_path)
    ####
    def load_checkpoint(self):

        encoderX_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderX.pt' % self.ckpt_load_iter
        )
        encoderY_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_encoderY.pt' % self.ckpt_load_iter
        )
        decoderY_path = os.path.join(
            self.ckpt_dir,
            'iter_%s_decoderY.pt' % self.ckpt_load_iter
        )

        if self.device == 'cuda':
            self.encoderX = torch.load(encoderX_path)
            self.encoderY = torch.load(encoderY_path)
            self.decoderY = torch.load(decoderY_path)
        else:
            self.encoderX = torch.load(encoderX_path, map_location='cpu')
            self.encoderY = torch.load(encoderY_path, map_location='cpu')
            self.decoderY = torch.load(decoderY_path, map_location='cpu')
