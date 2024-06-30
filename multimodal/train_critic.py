"""
    Train the Trajectory Scoring Module only
"""

import os
import time
import sys
import shutil 
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data
from PIL import Image
from subprocess import call
import utils
from multimodaldataset import multimodaldata
import ipdb


def train(conf):
    # create training and validation datasets and data loaders
    data_features = ['pc', 'franka_state', 'action', 'rotmat', 'critic_score', 'hand_pos', 'root_pos']
     
    # load network model
    model_def = utils.get_model_module(conf.model_version)

    # create models
    network = model_def.CriticNetwork(pc_input_feat = conf.pc_input_feat, pc_feat_dim = conf.pc_feat_dim, state_feat_dim=conf.state_feat_dim, 
                                      action_feat_dim=conf.action_feat_dim, action_dim=conf.action_dim, state_dim=conf.state_dim, more = True)
    
    if conf.continue_to_play:
        network.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-network.pth' % conf.saved_proposal_epoch)))

    # create optimizers
    network_opt = torch.optim.Adam(network.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    # learning rate scheduler
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(network_opt, step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)

    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch     Dataset     LR    TotalLoss'
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pytorch
        # from tensorboardX import SummaryWriter
        tb_writer = SummaryWriter(conf.tb_dir)

    # send parameters to device
    network.to(conf.device)
    utils.optimizer_to_device(network_opt, conf.device)
    print("network and optimizer config complete!!")

    train_dataset = multimodaldata(conf, data_features, train_mode=True)
    train_dataset.load_process_data(conf.process_data_path)
    train_dataset.process_data()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, \
                                                   num_workers=0, drop_last=True)
    train_num_batch = len(train_dataloader)

    val_dataset = multimodaldata(conf, data_features, train_mode=False)
    val_dataset.load_process_data(conf.process_data_path)
    val_dataset.process_data()
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True, \
                                                   num_workers=0, drop_last=True)
    val_num_batch = len(val_dataloader)
    print('train_num_batch: %d' % (train_num_batch))
    print('val_num_batch: %d' % (val_num_batch))
    
    # start training
    start_time = time.time()

    start_epoch = 0
    if conf.continue_to_play:
        start_epoch = conf.saved_critic_epoch

    # train for every epoch
    for epoch in range(start_epoch, conf.epochs):

        ### print log
        if not conf.no_console_log:
            utils.printout(conf.flog, f'training run {conf.exp_name}')
            utils.printout(conf.flog, header)

        train_batches = enumerate(train_dataloader, 0)
        val_batches = enumerate(val_dataloader, 0)

        train_fraction_done = 0.0
        val_fraction_done = 0.0
        val_batch_ind = -1

        train_epoch_loss = {}
        train_epoch_loss['tot'] = 0.0
        val_epoch_loss = {}
        val_epoch_loss['tot'] = 0.0
        ep_cnt = 0

        ### train for every batch
        for train_batch_ind, batch in train_batches:
            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind

            log_console = not conf.no_console_log 
            
            # save checkpoint
            if epoch % 2 == 0 and train_batch_ind == 0:
                with torch.no_grad():
                    utils.printout(conf.flog, 'Saving checkpoint ...... ')
                    torch.save(network.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-network.pth' % epoch))
                    torch.save(network_opt.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-optimizer.pth' % epoch))
                    torch.save(network_lr_scheduler.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-lr_scheduler.pth' % epoch))
                    # torch.save(train_dataset, os.path.join(conf.exp_dir, 'ckpts', '%d-train_dataset.pth' % epoch))
                    utils.printout(conf.flog, 'DONE')

            # set models to training mode
            network.train()

            losses = critic_forward(batch=batch, data_features=data_features, network=network, device=conf.device)

            total_loss = losses['tot']
            
            # optimize one step
            network_opt.zero_grad()
            total_loss.backward()
            network_opt.step()
            network_lr_scheduler.step()
            for key in train_epoch_loss.keys():
                train_epoch_loss[key] += losses[key]
            
            # validate one batch
            val_cnt = 0
            while val_fraction_done <= train_fraction_done and val_batch_ind+1 < val_num_batch:
                val_cnt += 1
                val_batch_ind, val_batch = next(val_batches)

                val_fraction_done = (val_batch_ind + 1) / val_num_batch
                val_step = (epoch + val_fraction_done) * train_num_batch - 1

                # set models to evaluation mode
                network.eval()

                with torch.no_grad():
                    # forward pass (including logging)
                    losses = critic_forward(batch=batch, data_features=data_features, network=network, device=conf.device)
                for key in val_epoch_loss.keys():
                    val_epoch_loss[key] += losses[key]
        ep_cnt += 1
        for key in train_epoch_loss.keys():
            train_epoch_loss[key] /= (ep_cnt * train_num_batch)
            val_epoch_loss[key] /= (ep_cnt * val_num_batch)
        if log_console:
            data_split = 'train'
            lr = network_opt.param_groups[0]['lr']
            utils.printout(conf.flog, \
                           f'''{strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} '''
                           f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                           f'''{data_split:^10s} '''
                           f'''{lr:>5.2E} '''
                           f'''{train_epoch_loss['tot']:>10.5f}'''
                           )
            data_split = 'val'
            utils.printout(conf.flog, \
                           f'''{strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} '''
                           f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                           f'''{data_split:^10s} '''
                           f'''{lr:>5.2E} '''
                           f'''{val_epoch_loss['tot']:>10.5f}'''
                           )
            conf.flog.flush()

        # log to tensorboard
        if not conf.no_tb_log and tb_writer is not None:
            tb_writer.add_scalar('tot_loss/train', train_epoch_loss['tot'], epoch)
            
            tb_writer.add_scalar('tot_loss/val', val_epoch_loss['tot'], epoch)
            
            tb_writer.add_scalar('lr', lr, epoch)


def critic_forward(batch, data_features, network, device):
    # prepare input
    pc = batch[data_features.index('pc')].to(device)  # B x 4096 x 3
    franka_state = batch[data_features.index('franka_state')].to(device)
    hand_pos = batch[data_features.index('hand_pos')].to(device)
    root_pos = batch[data_features.index('root_pos')].to(device)

    # ik = batch[data_features.index('ik')].to(device)
    action = batch[data_features.index('action')].to(device)
    rotmat = batch[data_features.index('rotmat')].to(device)  # B x 10 x 3

    action_9d = torch.cat([action, rotmat[:, :2, :].reshape(-1, 6)], dim=-1)
    gt_score = batch[data_features.index('critic_score')].to(device)
    losses = network.get_loss(pc, franka_state[:,:18], action_9d, gt_score.squeeze(-1), hand_pos, root_pos, more=True)

    return losses


if __name__ == '__main__':
    ### get parameters
    parser = ArgumentParser()
    
    # main parameters (required)
    parser.add_argument('--exp_suffix', type=str, default='lever_pull', help='exp suffix')
    parser.add_argument('--train_mode', action='store_true', default=False)
    parser.add_argument('--model_version', type=str, default='model_critic', help='model def file')
    parser.add_argument('--device_id', type=int, default=2, help='cpu or cuda:x for using cuda on GPU number x')
    
    parser.add_argument('--data_path', type=str, default='../logs/franka_slider_door', help='data path')
    parser.add_argument('--data_num', type=int, default=11, help='data num')
    parser.add_argument('--process_data_path', type=str, default='../logs/process_data/open_lever_pull_right_critic_multistep', help='process data path')

    # main parameters (optional)
    parser.add_argument('--seed', type=int, default=-1, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='../logs', help='exp logs directory')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if exp_dir exists [default: False]')
    parser.add_argument('--resume', action='store_true', default=False, help='resume if exp_dir exists [default: False]')
    # training parameters
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=5000)
    parser.add_argument('--saved_proposal_epoch', type=int, default=500)

    parser.add_argument('--pc_feat_dim', type=int, default=128, help='pointnet++ dim of network')
    parser.add_argument('--state_feat_dim', type=int, default=64, help='feature dim of critic network')
    parser.add_argument('--action_feat_dim', type=int, default=64, help='feature dim of critic network')
    parser.add_argument('--pc_input_feat', type=int, default=3, help='feature dim of critic network')
    parser.add_argument('--action_dim', type=int, default=9, help='action representation dim')
    parser.add_argument('--state_dim', type=int, default=18, help='state representation dim')
    parser.add_argument('--residual', action='store_true', default=False, help='action representation dim')
    parser.add_argument('--only_one_step', action='store_true', default=True, help='action representation dim')
    parser.add_argument('--open_door_multistep', action='store_true', default=False, help='open_door_multistep')
    parser.add_argument('--pointnum', type=int, default=4096, help='action representation dim')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='action representation dim')
    parser.add_argument('--ik_dim', type=int, default=9, help='action ik representation dim')
    parser.add_argument('--ik_feat_dim', type=int, default=64, help='action ik feat representation dim')
    # loss weights

    # logging
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=10, help='number of optimization steps beween console log prints')

    # continue to play
    parser.add_argument('--continue_to_play', action='store_true', default=False, help='continue to play')

    # finetune
    parser.add_argument('--finetune', action='store_true', default=False)

    # parse args
    conf = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(conf.device_id)
    
    conf.device = "cuda:%d"%(conf.device_id)
    print(conf.device)

    # make exp_name
    conf.exp_name = f'exp-{conf.model_version}-{conf.exp_suffix}'

    if conf.overwrite and conf.resume:
        raise ValueError('ERROR: cannot specify both --overwrite and --resume!')

    # mkdir exp_dir; ask for overwrite if necessary; or resume
    conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)
    print('exp_dir: ', conf.exp_dir)
    conf.tb_dir = os.path.join(conf.exp_dir, 'tb')
    if os.path.exists(conf.exp_dir):
        if not conf.resume:
            if not conf.overwrite:
                response = input('A training run named "%s" already exists, overwrite? (y/n) ' % conf.exp_name)
                if response != 'y':
                    exit(1)
            shutil.rmtree(conf.exp_dir)
    else:
        if conf.resume:
            raise ValueError('ERROR: no training run named %s to resume!' % conf.exp_name)
    if not conf.resume:
        os.mkdir(conf.exp_dir)
        os.mkdir(conf.tb_dir)
        os.mkdir(os.path.join(conf.exp_dir, 'ckpts'))
        os.mkdir(os.path.join(conf.exp_dir, 'pred_score'))


    # file log
    if conf.resume:
        flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'a+')
    else:
        flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'w')
    conf.flog = flog
    train(conf)