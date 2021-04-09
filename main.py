import argparse
import numpy as np
import torch

#-----------------------------------------------------------------------------#
from data.loader import data_loader
from solver import Solver
from utils import str2bool, bool_flag
import os
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

#-----------------------------------------------------------------------------#
    
def create_parser():
    
    '''
    Create a parser for command-line arguments
    '''
    
    parser = argparse.ArgumentParser()

    parser.add_argument( '--run_id', default=42, type=int,
      help='run id (default=-1 to create a new id)' )

    parser.add_argument( '--device', default='cpu', type=str,
      help='cpu/cuda' )

    # training hyperparameters
    parser.add_argument( '--batch_size', default=64, type=int,
      help='batch size' )
    parser.add_argument( '--lr_VAE', default=1e-3, type=float,
      help='learning rate of the VAE' )
    parser.add_argument( '--beta1_VAE', default=0.9, type=float,
      help='beta1 parameter of the Adam optimizer for the VAE' )
    parser.add_argument( '--beta2_VAE', default=0.999, type=float,
      help='beta2 parameter of the Adam optimizer for the VAE' )




    # saving directories and checkpoint/sample iterations
    parser.add_argument( '--ckpt_load_iter', default=0, type=int,
      help='iter# to load the previously saved model ' +
        '(default=0 to start from the scratch)' )
    parser.add_argument( '--max_iter', default=10, type=float,
      help='maximum number of batch iterations' )
    parser.add_argument( '--ckpt_save_iter', default=100, type=int,
      help='checkpoint saved every # iters' )
    parser.add_argument( '--output_save_iter', default=10000, type=int,
      help='output saved every # iters' )
    parser.add_argument( '--print_iter', default=10, type=int,
      help='print losses iter' )
    # parser.add_argument( '--eval_metrics_iter', default=50, type=int,
    #   help='evaluate metrics every # iters' )

    # visdom
    parser.add_argument( '--viz_on',
      default=False, type=str2bool, help='enable visdom visualization' )
    parser.add_argument( '--viz_port',
      default=8002, type=int, help='visdom port number' )
    parser.add_argument( '--viz_ll_iter',
      default=30, type=int, help='visdom line data logging iter' )
    parser.add_argument( '--viz_la_iter',
      default=30, type=int, help='visdom line data applying iter' )
    #parser.add_argument( '--viz_ra_iter',
    #  default=10000, type=int, help='visdom recon image applying iter' )
    #parser.add_argument( '--viz_ta_iter',
    #  default=10000, type=int, help='visdom traverse applying iter' )


    # Dataset options
    parser.add_argument('--delim', default='tab', type=str)
    parser.add_argument('--loader_num_workers', default=4, type=int)
    parser.add_argument('--obs_len', default=8, type=int)
    parser.add_argument('--pred_len', default=12, type=int)
    parser.add_argument('--skip', default=1, type=int)
    # dataset
    parser.add_argument( '--dataset_dir', default='../datasets', type=str,
      help='dataset directory' )
    parser.add_argument( '--dataset_name', default='hotel', type=str,
      help='dataset name' )
    parser.add_argument( '--num_workers', default=0, type=int,
      help='dataloader num_workers' )




    # model hyperparameters
    parser.add_argument( '--zS_dim', default=64, type=int,
      help='dimension of the shared latent representation' )
    # Encoder
    parser.add_argument('--encoder_h_dim', default=32, type=int)
    parser.add_argument('--decoder_h_dim', default=128, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--dropout_mlp', default=0.1, type=float)
    parser.add_argument('--dropout_rnn', default=0.25, type=float)
    # Decoder
    parser.add_argument('--pool_every_timestep', default=0, type=bool_flag)
    parser.add_argument('--mlp_dim', default=32, type=int)
    parser.add_argument('--batch_norm', default=0, type=bool_flag)

    parser.add_argument( '--attention', default=0, type=bool_flag,
      help='pool/attn' )
    parser.add_argument( '--kl_weight', default=100.0, type=float,
      help='kl weight' )
    parser.add_argument('--map_size', default=160, type=int)

    parser.add_argument( '--desc', default='data', type=str,
      help='run description' )
    return parser

#-----------------------------------------------------------------------------#

def main(args):


    if args.ckpt_load_iter == args.max_iter:

        print("Initializing test dataset")
        if args.dataset_name=='all':
            print('======================== [iter_%d] ========================' %  args.ckpt_load_iter)
            for dataset_name in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
                args.dataset_name =dataset_name

                solver = Solver(args)
                test_path = os.path.join(args.dataset_dir, dataset_name, 'test')
                _, test_loader = data_loader(args, test_path)

                ade_min, fde_min, \
                ade_avg, fde_avg, \
                ade_std, fde_std = solver.evaluate_dist(test_loader, 20, loss=False)
                print(args.dataset_name)
                print('ade min: ', ade_min)
                print('ade avg: ', ade_avg)
                print('ade std: ', ade_std)
                print('fde min: ', fde_min)
                print('fde avg: ', fde_avg)
                print('fde std: ', fde_std)
        else:
            solver = Solver(args)

            # dist_path = os.path.join(args.dataset_dir, args.dataset_name, 'dist')
            # print(dist_path)
            # _, dist_loader = data_loader(args, dist_path)
            # solver.check_dist_stat(dist_loader)


            test_path = os.path.join(args.dataset_dir, args.dataset_name, 'test')
            args.batch_size=364
            _, test_loader = data_loader(args, test_path,shuffle=False)
            solver.plot_traj_var(test_loader)
            # solver.draw_traj(test_loader, 20)
            # solver.check_dist_stat(test_loader)


            ade_min, fde_min, \
            ade_avg, fde_avg, \
            ade_std, fde_std = solver.evaluate_dist(test_loader, 20, loss=False)
            print('--------------------', args.dataset_name , '----------------------')
            print('ade min: ', ade_min)
            print('ade avg: ', ade_avg)
            print('ade std: ', ade_std)
            print('fde min: ', fde_min)
            print('fde avg: ', fde_avg)
            print('fde std: ', fde_std)
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
