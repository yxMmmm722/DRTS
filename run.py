import argparse
import os
import random

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args


def build_parser():
    parser = argparse.ArgumentParser(description='DRTS long-term forecasting')

    # Run config
    parser.add_argument('--is_training', type=int, default=1, help='1: train and test, 0: test from checkpoint')
    parser.add_argument('--model_id', type=str, default='DRTS', help='experiment id prefix')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # Data
    parser.add_argument('--data', type=str, required=True, help='dataset type: ETTh1/ETTh2/ETTm1/ETTm2/custom')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', choices=['M', 'S', 'MS'])
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='frequency for time feature encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='checkpoint directory')

    # Forecast task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--inverse', action='store_true', default=False, help='save predictions in original scale')

    # DRTS model
    parser.add_argument('--enc_in', type=int, default=7, help='number of input variables')
    parser.add_argument('--n_period', type=int, default=1, help='number of retrieval scales')
    parser.add_argument('--topm', type=int, default=20, help='top-k historical windows for retrieval')
    parser.add_argument('--decomp_kernel', type=int, default=25, help='STL trend smoother/window')
    parser.add_argument('--seasonal_kernel', type=int, default=8, help='STL seasonal smoother/window')
    parser.add_argument('--stl_period', type=int, default=0, help='STL seasonal period; 0 infers from dataset')
    parser.add_argument('--component_weights', type=str, default='0.4,0.4,0.2',
                        help='fixed fusion weights for trend,seasonal,residual references')
    parser.add_argument('--spatial_top_m', type=int, default=3, help='top-m variables kept in dynamic spatial graph')
    parser.add_argument('--retrieval_batch_size', type=int, default=0,
                        help='batch size for retrieval precomputation; 0 uses a safe default')
    parser.add_argument('--component_lazy_full_series', type=int, default=0,
                        help='keep STL full-series cache and slice windows lazily')
    parser.add_argument('--use_benefit_scorer', type=int, default=0,
                        help='1: train a supervised benefit scorer to weight retrieved candidates')
    parser.add_argument('--benefit_base_epochs', type=int, default=1,
                        help='base-only warmup epochs before fitting the benefit scorer')
    parser.add_argument('--benefit_scorer_batch_size', type=int, default=0,
                        help='batch size for fitting the benefit scorer; 0 uses retrieval_batch_size/default')
    parser.add_argument('--benefit_scorer_ridge', type=float, default=1e-3,
                        help='ridge regularization for the linear benefit scorer')
    parser.add_argument('--benefit_scorer_max_batches', type=int, default=0,
                        help='optional cap on batches used to fit the benefit scorer; 0 uses all')
    parser.add_argument('--retrieval_diversity_delta', type=int, default=0,
                        help='minimum index gap between retrieved windows; 0 disables diversity')
    parser.add_argument('--retrieval_diversity_mult', type=int, default=3,
                        help='candidate multiplier before diverse top-k selection')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', choices=['type1', 'type2', 'cosine'],
                        help='learning-rate schedule')
    parser.add_argument('--use_amp', action='store_true', default=False, help='use automatic mixed precision')

    # Device
    parser.add_argument('--use_gpu', type=int, default=1, help='1: use CUDA when available, 0: force CPU')
    parser.add_argument('--gpu', type=int, default=0, help='main GPU id')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple GPUs')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='multi-GPU device ids')
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_stl_period(args):
    if args.stl_period > 0:
        return

    data_path_key = os.path.splitext(os.path.basename(args.data_path))[0].lower()
    if args.data in ['ETTh1', 'ETTh2'] or data_path_key in ['etth1', 'etth2']:
        args.stl_period = 24
    elif args.data in ['ETTm1', 'ETTm2'] or data_path_key in ['ettm1', 'ettm2']:
        args.stl_period = 96
    elif 'weather' in data_path_key:
        args.stl_period = 144
    elif 'electricity' in data_path_key or 'traffic' in data_path_key:
        args.stl_period = 24
    elif 'exchange' in data_path_key:
        args.stl_period = 7
    elif 'illness' in data_path_key:
        args.stl_period = 52
    else:
        args.stl_period = max(2, args.seasonal_kernel)


def configure_device(args):
    args.use_gpu = bool(args.use_gpu and torch.cuda.is_available())
    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]


def build_setting(args):
    weights = args.component_weights.replace(',', '-')
    benefit_suffix = f'_ben{args.use_benefit_scorer}' if int(args.use_benefit_scorer) else ''
    return (
        f'long_term_forecast_{args.model_id}_{args.data}'
        f'_ft{args.features}_sl{args.seq_len}_pl{args.pred_len}'
        f'_np{args.n_period}_topm{args.topm}_stl{args.stl_period}_cw{weights}'
        f'{benefit_suffix}'
    )


if __name__ == '__main__':
    args = build_parser().parse_args()
    set_seed(args.seed)
    infer_stl_period(args)
    configure_device(args)

    print('Args in experiment:')
    print_args(args)

    exp = Exp_Long_Term_Forecast(args)
    setting = build_setting(args)

    if args.is_training:
        print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train(setting)

        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting)
    else:
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)

    torch.cuda.empty_cache()
