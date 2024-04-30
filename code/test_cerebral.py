import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import numpy as np
import data
import data2
from data import TwoStreamBatchSampler ,relabel_dataset
import shutil
from trainval_classifier import train_casenet ,val_casenet,val_casenet_1
from utils import Logger ,save_itk,weights_init,exp_warmup
from networks.ResNet34 import Resnet34

from split_combine_mj import SplitComb
import torch.optim.lr_scheduler as lr_scheduler
import torch
from torch.nn import DataParallel
import torch.nn as nn
import glob
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
import csv
from option import parser
import  sys
from test import evaluation,evaluation_1
from tensorboardX import SummaryWriter
from importlib import import_module
from networks.vnet import VNet
# import nibabel
from networks.ResNet34 import Resnet34
from networks.model_mj_in3sad import UNet3D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
parsing_path = r"D:\daipeng\supervised-full\tree_pase_random"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


global args
args = parser.parse_args()
def main():
    MODE = 'test'
    print('start testing')
    start_epoch = args.start_epoch
    save_dir = args.save_dir
    save_dir = os.path.join(r'D:\daipeng\semi_aricerebral\ours\code\train_MCF_airway\2023__semi_MCF+reliable_label_2_mse', save_dir)
    print("savedir:", save_dir)

    marginv = args.cubesize
    model = import_module(args.model)
    # ema_model_  = import_module(args.model)
    config, _ = model.get_model(args, False)
    margin = args.cubesize


    net = UNet3D(1, 2, False)
    # net = Resnet34(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True)
    model_resnet = VNet(n_channels=1, n_classes=2, normalization='batchnorm', n_filters=16,has_dropout=False)
    # net = nn.DataParallel(net).cuda()
    # model_resnet =nn.DataParallel(model_resnet).cuda()
    print('patch size', margin)
    print('train stride', args.stridet)

    split_comber = SplitComb(args.stridet, margin)
    print('-----------------------------------')
    split_comber = SplitComb(args.stridev, marginv)
    print('val patch size ', marginv)
    print('validation stride', args.stridev)
    # dataset_test = data2.val_dataset(
    #     config,
    #     args,
    #     split_comber=split_comber,
    #     debug=args.debug
    # )

    dataset_test = data2.test_dataset(
        config,
        args,
        split_comber=split_comber,
        debug=args.debug
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False
    )
    # epoch = 2901
    # epoch = 28
    epoch=20
    args.test_dir = os.path.join(save_dir, 'test%3d' % epoch)

    print(f"test_dir{args.test_dir}")
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    # net.load_state_dict(torch.load(r"D:\daipeng\MCF+reliable_label\code\train_MCF_airway\2023_3_24__semi_MCF+reliable_label\epoch_28.0000-mean_sensiti288.0476-dice90.9264")[
    #                         'state_dict'])

    net.load_state_dict(torch.load(r"D:\daipeng\semi_aricerebral\ours\code\train_MCF_airway\2023__semi_MCF+reliable_label_2_mse\epoch_60.0000-mean_sensiti265.6123-dice88.5476")[
                            'state_dict'])
    # model_resnet.load_state_dict(torch.load(r'D:\daipeng\MCF+reliable_label\code\train_MCF_airway\2023_3_22_2_semi_MCF+reliable_label\epoch_resnet30.0000-mean_sensiti285.8961-dice78.1577'
    #                                          )['state_dict'])
    #v_loss, mean_sensiti2, mean_dice2 = val_casenet(epoch,net,model_resnet,test_loader, args, MODE)
    v_loss, mean_sensiti2, mean_dice2 = val_casenet_1(epoch, net ,test_loader, args, MODE)
    evaluation_1(args.test_dir, parsing_path)

def temp_0673():
    path = r"C:\daipeng_airway_semi_contrast\2.MCF-main\MCF-main\code\train_MCF_airway\results_0.27_1_2_MCF\test_temp"
    evaluation(path,parsing_path)



if __name__=="__main__":
    main()