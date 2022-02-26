import argparse
import numpy as np
import torch

# -----------------------------------------------------------------------------#
from data.loader import data_loader
from solver import Solver
from utils import str2bool, bool_flag
import os

from data.nuscenes.config import Config
from data.nuscenes_dataloader import data_generator
###############################################################################

# set the random seed manually for reproducibility
SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


###############################################################################

def print_opts(opts):
    '''
    Print the values of all command-line arguments
    '''

    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


# -----------------------------------------------------------------------------#

def create_parser():
    '''
    Create a parser for command-line arguments
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_id', default=14, type=int,
                        help='run id (default=-1 to create a new id)')

    parser.add_argument('--device', default='cpu', type=str,
                        help='cpu/cuda')

    # training hyperparameters
    parser.add_argument('--batch_size', default=2, type=int,
                        help='batch size')
    parser.add_argument('--lr_VAE', default=1e-3, type=float,
                        help='learning rate of the VAE')
    parser.add_argument('--beta1_VAE', default=0.9, type=float,
                        help='beta1 parameter of the Adam optimizer for the VAE')
    parser.add_argument('--beta2_VAE', default=0.999, type=float,
                        help='beta2 parameter of the Adam optimizer for the VAE')

    # saving directories and checkpoint/sample iterations
    parser.add_argument('--ckpt_load_iter', default=0, type=int,
                        help='iter# to load the previously saved model ' +
                             '(default=0 to start from the scratch)')
    parser.add_argument('--max_iter', default=100, type=float,
                        help='maximum number of batch iterations')
    parser.add_argument('--ckpt_save_iter', default=100, type=int,
                        help='checkpoint saved every # iters')
    parser.add_argument('--output_save_iter', default=10000, type=int,
                        help='output saved every # iters')
    parser.add_argument('--print_iter', default=10, type=int,
                        help='print losses iter')
    # parser.add_argument( '--eval_metrics_iter', default=50, type=int,
    #   help='evaluate metrics every # iters' )

    # visdom
    parser.add_argument('--viz_on',
                        default=False, type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_port',
                        default=8002, type=int, help='visdom port number')
    parser.add_argument('--viz_ll_iter',
                        default=4, type=int, help='visdom line data logging iter')
    parser.add_argument('--viz_la_iter',
                        default=1, type=int, help='visdom line data applying iter')
    # parser.add_argument( '--viz_ra_iter',
    #  default=10000, type=int, help='visdom recon image applying iter' )
    # parser.add_argument( '--viz_ta_iter',
    #  default=10000, type=int, help='visdom traverse applying iter' )


    # Dataset options
    parser.add_argument('--delim', default=',', type=str)
    parser.add_argument('--loader_num_workers', default=0, type=int)
    parser.add_argument('--obs_len', default=8, type=int)
    parser.add_argument('--pred_len', default=12, type=int)
    # dataset
    # parser.add_argument('--dataset_dir', default='../datasets/Trajectories', type=str, help='dataset directory')
    # parser.add_argument('--dataset_dir', default='../datasets/SDD', type=str, help='dataset directory')
    # parser.add_argument('--dataset_dir', default='C:/dataset/KITTI-trajectory-prediction', type=str, help='dataset directory')
    parser.add_argument('--dataset_dir', default='C:\dataset\HTP-benchmark\A2A Data', type=str, help='dataset directory')
    parser.add_argument('--dataset_name', default='sdd.lgcvae', type=str,
                        help='dataset name')
    parser.add_argument('--model_name', default='', type=str,
                        help='dataset name')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='dataloader num_workers')

    # model hyperparameters
    parser.add_argument('--zS_dim', default=20, type=int,
                        help='dimension of the shared latent representation')
    # Encoder
    parser.add_argument('--encoder_h_dim', default=64, type=int)
    parser.add_argument('--decoder_h_dim', default=128, type=int)
    parser.add_argument('--map_feat_dim', default=32, type=int)

    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--dropout_mlp', default=0.3, type=float)
    parser.add_argument('--dropout_rnn', default=0.25, type=float)
    # Decoder
    parser.add_argument('--mlp_dim', default=128, type=int)
    parser.add_argument('--map_mlp_dim', default=128, type=int)
    parser.add_argument('--batch_norm', default=0, type=bool_flag)

    parser.add_argument('--kl_weight', default=100.0, type=float,
                        help='kl weight')
    parser.add_argument('--lg_kl_weight', default=0.05, type=float)

    parser.add_argument('--w_dim', default=10, type=int)
    parser.add_argument('--ll_prior_w', default=1.0, type=float)
    parser.add_argument('--no_convs_fcomb', default=2, type=int)
    parser.add_argument('--no_convs_per_block', default=1, type=int)
    parser.add_argument('--alpha', default=0.25, type=float)
    parser.add_argument('--gamma', default=2., type=float)
    parser.add_argument('--fb', default=0.5, type=float)
    parser.add_argument('--anneal_epoch', default=20, type=int)
    parser.add_argument('--aug', default=1, type=int)
    parser.add_argument('--load_e', default=3, type=int)
    parser.add_argument('--context_dim', default=10, type=int)
    parser.add_argument('--scale', default=1.0, type=float)
    parser.add_argument('--coll_th', default=5.0, type=float)
    parser.add_argument('--w_agent', default=1.0, type=float)
    parser.add_argument('--w_map', default=1.0, type=float)
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--lr_e', default=0.8, type=float)



    parser.add_argument('--desc', default='data', type=str,
                        help='run description')
    return parser


# -----------------------------------------------------------------------------#

def main(args):
    if args.ckpt_load_iter == args.max_iter:

        print("Initializing test dataset")
        solver = Solver(args)

        print('--------------------', args.dataset_name, '----------------------')
        args.batch_size = 4

        # cfg = Config('nuscenes', False, create_dirs=True)
        # torch.set_default_dtype(torch.float32)
        # log = open('log.txt', 'a+')
        # test_loader = data_generator(cfg, log, split='test', phase='testing',
        #                              batch_size=args.batch_size, device=args.device, scale=args.scale, shuffle=False)


        _, test_loader = data_loader(args, args.dataset_dir, 'test', shuffle=False)


        # solver.load_checkpoint()
        # solver.check_feat(test_loader)

        # solver.evaluate_lg(test_loader, num_gen=3)
        # solver.evaluate_each(test_loader)
        # solver.collision_stat(test_loader)
        solver.evaluate_dist(test_loader, loss=True)
        #
        # fde_min, fde_avg, fde_std = solver.evaluate_dist(test_loader, loss=False)
        # print(fde_min)
        # print(fde_avg)
        # print(fde_std)


        gh = True
        print("GEN HEAT MAP: ", gh)

        traj_path = 'sdd.traj_zD_20_dr_mlp_0.3_dr_rnn_0.25_enc_hD_64_dec_hD_128_mlpD_256_map_featD_32_map_mlpD_256_lr_0.001_klw_50.0_ll_prior_w_1.0_zfb_0.07_scale_100.0_num_sg_3_run_200'

        traj_iter = '15000'
        traj_ckpt = {'ckpt_dir': os.path.join('ckpts', traj_path), 'iter': traj_iter}
        print('===== TRAJECTORY:', traj_ckpt)

        # lg_path = 'lgcvae_enc_block_1_fcomb_block_2_wD_20_lr_0.001_lg_klw_1_a_0.25_r_2.0_fb_0.5_anneal_e_0_load_e_1_run_24'
        # lg_iter = '57100'

        lg_path = 'sdd.lgcvae_enc_block_1_fcomb_block_2_wD_20_lr_0.0001_lg_klw_1.0_a_0.25_r_2.0_fb_0.5_anneal_e_0_aug_1_run_181'
        lg_iter = '43000'
        lg_ckpt = {'ckpt_dir': os.path.join('ckpts', lg_path), 'iter': lg_iter}
        print('===== LG CVAE:', lg_ckpt)

        solver.pretrain_load_checkpoint(traj_ckpt, lg_ckpt)

        # solver.check_feat(test_loader)

        # solver.plot_traj_var(test_loader)
        # solver.evaluate_dist_gt_goal(test_loader)
        # solver.check_feat(test_loader)

        lg_num=20
        traj_num=1

        ade_min, fde_min, \
        ade_avg, fde_avg, \
        ade_std, fde_std, \
        sg_ade_min, sg_ade_avg, sg_ade_std, \
        lg_fde_min, lg_fde_avg, lg_fde_std = solver.all_evaluation(test_loader, lg_num=lg_num, traj_num=traj_num, generate_heat=gh)

        print('lg_num: ', lg_num, ' // traj_num: ', traj_num)
        print('ade min: ', ade_min)
        print('ade avg: ', ade_avg)
        print('ade std: ', ade_std)
        print('fde min: ', fde_min)
        print('fde avg: ', fde_avg)
        print('fde std: ', fde_std)
        print('sg_ade_min: ', sg_ade_min)
        print('sg_ade_avg: ', sg_ade_avg)
        print('sg_ade_std: ', sg_ade_std)
        print('lg_fde_min: ', lg_fde_min)
        print('lg_fde_avg: ', lg_fde_avg)
        print('lg_fde_std: ', lg_fde_std)
        print('------------------------------------------')

        lg_num = 10
        traj_num = 2

        ade_min, fde_min, \
        ade_avg, fde_avg, \
        ade_std, fde_std, \
        sg_ade_min, sg_ade_avg, sg_ade_std, \
        lg_fde_min, lg_fde_avg, lg_fde_std = solver.all_evaluation(test_loader, lg_num=lg_num, traj_num=traj_num, generate_heat=gh)

        print('lg_num: ', lg_num, ' // traj_num: ', traj_num)
        print('ade min: ', ade_min)
        print('ade avg: ', ade_avg)
        print('ade std: ', ade_std)
        print('fde min: ', fde_min)
        print('fde avg: ', fde_avg)
        print('fde std: ', fde_std)
        print('sg_ade_min: ', sg_ade_min)
        print('sg_ade_avg: ', sg_ade_avg)
        print('sg_ade_std: ', sg_ade_std)
        print('lg_fde_min: ', lg_fde_min)
        print('lg_fde_avg: ', lg_fde_avg)
        print('lg_fde_std: ', lg_fde_std)
        print('------------------------------------------')


        ade_min, fde_min, \
        ade_avg, fde_avg, \
        ade_std, fde_std, \
        sg_ade_min, sg_ade_avg, sg_ade_std, \
        lg_fde_min, lg_fde_avg, lg_fde_std = solver.evaluate_dist(test_loader, loss=False)

        print('ade min: ', ade_min)
        print('ade avg: ', ade_avg)
        print('ade std: ', ade_std)
        print('fde min: ', fde_min)
        print('fde avg: ', fde_avg)
        print('fde std: ', fde_std)
        print('sg_ade_min: ', sg_ade_min)
        print('sg_ade_avg: ', sg_ade_avg)
        print('sg_ade_std: ', sg_ade_std)
        print('lg_fde_min: ', lg_fde_min)
        print('lg_fde_avg: ', lg_fde_avg)
        print('lg_fde_std: ', lg_fde_std)
        print('------------------------------------------')


    else:
        solver = Solver(args)
        solver.train()


###############################################################################

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print_opts(args)

    main(args)
