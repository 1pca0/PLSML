import os
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import time
import numpy as np
from importlib import import_module
import torch
from torch.utils.data import DataLoader
import skimage.measure as measure
import  nibabel
from skimage.morphology import skeletonize_3d

# from sklearn.metrics import roc_curve
smooth = 1.
import SimpleITK as sitk
from split_combine_mj import SplitComb
import data
from trainval_classifier import train_casenet, val_casenet
from option import parser
from utils import Logger, save_itk, weights_init, debug_dataloader, exp_warmup
from torch.nn import DataParallel
from torch.backends import cudnn
import logging
from medpy import metric

import cv2
from scipy.ndimage.morphology import binary_dilation
from sklearn.metrics import roc_curve

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
parsing_path = os.path.join(BASE_DIR, 'tree_parse')
from tqdm import tqdm

from BD_TD  import post_process


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing


def save_itk(image, origin, spacing, filename):
    if type(origin) != tuple:
        origin = tuple(reversed(list(origin)))
    if type(spacing) != tuple:
        spacing = tuple(reversed(list(spacing)))
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, filename, True)


class Logger(object):
    """
    Logger from screen to txt file
    """

    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def dice_coef_np(y_pred, y_true):
    """
    :param y_pred: prediction
    :param y_true: target ground-truth
    :return: dice coefficient
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def ppv_np(y_pred, y_true):
    """
    :param y_pred: prediction
    :param y_true: target ground-truth
    :return: positive predictive value
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (np.sum(y_pred_f) + smooth)


def sensitivity_np(y_pred, y_true):
    """
    :param y_pred: prediction
    :param y_true: target ground-truth
    :return: sensitivity
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (np.sum(y_true_f) + smooth)


# def FPR_np(y_pred, y_true):
#     """
#     :param y_pred: prediction
#     :param y_true: target ground-truth
#     :return: sensitivity
#     """
#
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     fpr, tpr, thre = roc_curve(y_true_f, y_pred_f)
#     # print(fpr, tpr, thre)
#     index = list(thre).index(1)
#     FPR = fpr[index]
#     # sensitivity = TP/(TP+FN)
#     # specificity = TN + smooth / (np.sum(1 - y_true_f) + smooth)
#
#     # intersection = np.sum(y_true_f * y_pred_f)
#     return FPR

def network_prediction(args):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    save_dir = args.save_dir
    save_dir = r'C:\daipeng_airway_semi_contrast\Meanteacher\Vnet\code\train_Vnet_airway\new_way_mean_teacher_1_7'
    print("savedir: ", save_dir)
    logfile = os.path.join(save_dir, 'log.txt')
    sys.stdout = Logger(logfile)

    print('----------------------Load Model------------------------')
    model = import_module(args.model)
    config, net = model.get_model(args)

    if args.cubesizev is not None:
        marginv = args.cubesizev
    else:
        marginv = args.cubesize

    if args.resume:
        resume_part = args.resumepart
        checkpoint = torch.load(args.resume)

        if resume_part:
            """
            load part of the weight parameters
            """
            net.load_state_dict(checkpoint['state_dict'], strict=False)
            print('part load Done')
        else:
            """
            load full weight parameters
            """
            net.load_state_dict(checkpoint['state_dict'])
            print("full resume Done")
    else:
        weights_init(net, init_type='xavier')  # weight initialization

    net = net.cuda()
    cudnn.benchmark = True
    if args.multigpu:
        net = DataParallel(net)

    print('---------------------testing---------------------')
    split_comber = SplitComb(args.stridev, marginv)
    dataset_test = data.val_dataset(
        config,
        args,
        split_comber=split_comber,
        debug=args.debug)
    test_loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)
    epoch = args.epochs
    MODE = 'test'
    print('start testing')
    args.test_dir = os.path.join(save_dir, 'test%.3d' % epoch)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    v_loss, mean_sensiti2, mean_dice2 = val_casenet(epoch, net, test_loader, args, MODE)

    return args.test_dir


def evaluation(data_path, parsing_path):
    logging.basicConfig(filename= r"2023_1-7_new_way_mean_teacher",level=logging.INFO)

    print("你进来了吗")
    file_list = os.listdir(data_path)
    file_list.sort()
    file_list_parse = os.listdir(parsing_path)
    file_list_parse.sort()
    n = 2
    if torch.cuda.is_available():
        device = torch.device("cuda")
    sens = []
    pres = []
    branches = []
    dice = []
    se = []
    fpr = []
    dice2 = []
    hd95 = []
    assd = []
    jc = []
    for i in tqdm(range(len(file_list) // n)):
        name = file_list[n * i]
        # img = nibabel.load(os.path.join(data_path, file_list[n*i]))

        label = nibabel.load(os.path.join(data_path, file_list[n * i]))
        # label, origin, spacing = load_itk_image(os.path.join(data_path, file_list[n*i+1]))
        pred2 = nibabel.load(os.path.join(data_path, file_list[n * i + 1]))


        # pred, origin, spacing = load_itk_image(os.path.join(data_path, file_list[n*i+2]))
        # 这里由于parsing_path 是120个拆分开的 这里需要修改 进行 寻找到那个对应的parse_path
        for  parse in file_list_parse:
            if file_list[n*i].split('-')[2] in parse:
                if 'parse' in parse:
                    parsing = nibabel.load(os.path.join(parsing_path, parse))  # please refer to tree_parse.py
        # parsing, origin, spacing = load_itk_image(os.path.join(parsing_path, file_list_parse[6*i+4]))
        # img = img.get_data()

        label = label.get_fdata()
        # label = torch.tensor(label).to(device)

        pred2 = pred2.get_fdata()

        # print("post_process")
        # seg_final, airway_prob_map = post_process(pred2, 0.5, False, True, 3, device)
        # pred2 = seg_final

        dice_metric = metric.binary.dc(pred2,label)
        hd95_metric = metric.binary.hd95(pred2,label)
        jc_metric  = metric.binary.jc(pred2,label)
        assd_metric = metric.binary.assd(pred2,label)

        dice2.append(dice_metric)
        hd95.append(hd95_metric)
        jc.append(jc_metric)
        assd.append(assd_metric)

        print(f"pred2{pred2.shape}")
        # kernel_3d = np.ones((3, 3, 3), np.uint8)
        # # 执行三维膨胀操作
        # dilated_3d_data = binary_dilation(pred2, kernel_3d, iterations=1)
        # pred = dilated_3d_data
        # pred = torch.tensor(pred).to(device)

        pred = pred2
        parsing = parsing.get_fdata()
        # parsing = torch.tensor(parsing).to(device)
        # parsing = parsing.transpose(2, 1, 0)

        curdice = dice_coef_np(pred, label)
        dice.append(curdice)
        cursensi = sensitivity_np(pred, label)
        se.append(cursensi)
        # curfpr = FPR_np(pred, label)
        # fpr.append(curfpr)

        cd, num = measure.label(pred, return_num=True, connectivity=1)
        cd_tensor = torch.tensor(cd, dtype=torch.int32).to('cuda')
        volume_tensor = torch.zeros(num, dtype=torch.int32).to("cuda")
        volume = np.zeros([num])
        for k in range(num):
            volume_tensor[k] = (cd_tensor == (k + 1)).sum()

        volume_result = volume_tensor.cpu().numpy()
        volume_sort = np.argsort(volume_result)
        # print(volume_sort)
        # if name == 'LIDC-IDRI-0429_img.nii.gz' or name == 'LIDC-IDRI-0837_img.nii.gz':
        #     large_cd = (cd == (volume_sort[-2] + 1)).astype(np.uint8)
        # else:
        #     large_cd = (cd==(volume_sort[-1]+1)).astype(np.uint8)
        large_cd = (cd == (volume_sort[-1] + 1)).astype(np.uint8)

        skeleton = skeletonize_3d(label)
        skeleton = (skeleton > 0)
        skeleton = skeleton.astype('uint8')

        sen = (large_cd * skeleton).sum() / skeleton.sum()
        sens.append(sen)

        pre = (large_cd * label).sum() / large_cd.sum()
        pres.append(pre)

        # print("能到这，啊")
        num_branch = parsing.max()
        detected_num = 0
        print(f"num_branch{num_branch}")
        for j in range(int(num_branch)):
            branch_label = ((parsing == (j + 1)).astype(np.uint8)) * skeleton
            if (large_cd * branch_label).sum() / branch_label.sum() >= 0.8:
                detected_num += 1
        branch = detected_num / num_branch
        branches.append(branch)

        print(file_list[n * i].split('_')[0], "Length: %0.4f" % (sen), "Precision: %0.4f" % (pre),
              "Branch: %0.4f" % (branch), "Dice: %0.4f" % (curdice), "Se: %0.4f" % (cursensi))

        name =file_list[n * i].split('_')[0]
        logging.info(f'name:{name},length: :{sen:.4f},precision :{pre:.4f},Branch:{branch:.4f},Dice:{curdice:.4f},Se{cursensi:.4f}')
        logging.info(f"---------------------------")
        logging.info(f" 尝试思路如下  ")


    sen1_mean = np.mean(sens)
    sen1_std = np.std(sens)
    pre_mean = np.mean(pres)
    pre_std = np.std(pres)
    branch_mean = np.mean(branches)
    branch_std = np.std(branches)
    dice_mean = np.mean(dice)
    se_mean = np.mean(se)
    dice2_mean = np.mean(dice2)
    jc_mean = np.mean(jc_metric)
    assd_mean = np.mean(assd_metric)
    hd95_mean  = np.mean(hd95_metric)
    # fpr_mean = np.mean(fpr)
    logging.info(
        f'sum,length: :{sen:.4f},precision :{pre:.4f},Branch:{branch:.4f},Dice:{curdice:.4f},Se{cursensi:.4f}')
    print("len mean: %0.4f (%0.4f), branch: %0.4f (%0.4f), pre: %0.4f (%0.4f), dice: %0.4f, se: %0.4f" % (
    sen1_mean, sen1_std, branch_mean, branch_std, pre_mean, pre_std, dice_mean, se_mean))
    print(f"dice2:{dice2_mean:.4f},jc_mean:{jc_mean:.4f},assd_mean:{assd_mean:.4f},hd95_mean:{hd95_mean:.4f}")
def FPR_np(y_pred, y_true):
    """
    :param y_pred: prediction
    :param y_true: target ground-truth
    :return: sensitivity
    """

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    fpr, tpr, thre = roc_curve(y_true_f, y_pred_f)
    # print(fpr, tpr, thre)
    index = list(thre).index(1)
    FPR = fpr[index]
    TPR = tpr[index]
    # sensitivity = TP/(TP+FN)
    # specificity = TN + smooth / (np.sum(1 - y_true_f) + smooth)

    # intersection = np.sum(y_true_f * y_pred_f)
    return FPR,TPR
def evaluation_1(data_path, parsing_path):
    logging.basicConfig(filename= r"2023_1-7_new_way_mean_teacher",level=logging.INFO)

    print("你进来了吗")
    file_list = os.listdir(data_path)
    file_list.sort()
    file_list_parse = os.listdir(parsing_path)
    file_list_parse.sort()
    n = 2
    if torch.cuda.is_available():
        device = torch.device("cuda")
    sens = []
    pres = []
    branches = []
    dice = []
    se = []
    TPR = []
    FPR = []
    dice2 = []
    hd95 = []
    assd = []
    jc = []
    for i in tqdm(range(len(file_list) // n)):
        name = file_list[n * i]
        # img = nibabel.load(os.path.join(data_path, file_list[n*i]))

        label = nibabel.load(os.path.join(data_path, file_list[n * i]))
        # label, origin, spacing = load_itk_image(os.path.join(data_path, file_list[n*i+1]))
        pred2 = nibabel.load(os.path.join(data_path, file_list[n * i + 1]))


        # pred, origin, spacing = load_itk_image(os.path.join(data_path, file_list[n*i+2]))
        # 这里由于parsing_path 是120个拆分开的 这里需要修改 进行 寻找到那个对应的parse_path
        for  parse in file_list_parse:
            if file_list[n*i].split('-')[2] in parse:
                if 'parse' in parse:
                    parsing = nibabel.load(os.path.join(parsing_path, parse))  # please refer to tree_parse.py
        # parsing, origin, spacing = load_itk_image(os.path.join(parsing_path, file_list_parse[6*i+4]))
        # img = img.get_data()

        label = label.get_fdata()
        # label = torch.tensor(label).to(device)

        pred2 = pred2.get_fdata()

        # print("post_process")
        # seg_final, airway_prob_map = post_process(pred2, 0.5, False, True, 3, device)
        # pred2 = seg_final

        dice_metric = metric.binary.dc(pred2,label)
        hd95_metric = metric.binary.hd95(pred2,label)
        jc_metric  = metric.binary.jc(pred2,label)
        assd_metric = metric.binary.assd(pred2,label)
        fpr, tpr = FPR_np(pred2, label)
        dice2.append(dice_metric)
        hd95.append(hd95_metric)
        jc.append(jc_metric)
        assd.append(assd_metric)
        FPR.append(fpr)
        TPR.append(tpr)

        print(f"pred2{pred2.shape}")
        # kernel_3d = np.ones((3, 3, 3), np.uint8)
        # # 执行三维膨胀操作
        # dilated_3d_data = binary_dilation(pred2, kernel_3d, iterations=1)
        # pred = dilated_3d_data
        # pred = torch.tensor(pred).to(device)

        pred = pred2
        parsing = parsing.get_fdata()
        # parsing = torch.tensor(parsing).to(device)
        # parsing = parsing.transpose(2, 1, 0)

        curdice = dice_coef_np(pred, label)
        dice.append(curdice)
        cursensi = sensitivity_np(pred, label)
        se.append(cursensi)
        # curfpr = FPR_np(pred, label)
        # fpr.append(curfpr)

        cd, num = measure.label(pred, return_num=True, connectivity=1)
        cd_tensor = torch.tensor(cd, dtype=torch.int32).to('cuda')
        volume_tensor = torch.zeros(num, dtype=torch.int32).to("cuda")
        volume = np.zeros([num])
        for k in range(num):
            volume_tensor[k] = (cd_tensor == (k + 1)).sum()

        volume_result = volume_tensor.cpu().numpy()
        volume_sort = np.argsort(volume_result)
        # print(volume_sort)
        # if name == 'LIDC-IDRI-0429_img.nii.gz' or name == 'LIDC-IDRI-0837_img.nii.gz':
        #     large_cd = (cd == (volume_sort[-2] + 1)).astype(np.uint8)
        # else:
        #     large_cd = (cd==(volume_sort[-1]+1)).astype(np.uint8)
        large_cd = (cd == (volume_sort[-1] + 1)).astype(np.uint8)

        skeleton = skeletonize_3d(label)
        skeleton = (skeleton > 0)
        skeleton = skeleton.astype('uint8')

        sen = (large_cd * skeleton).sum() / skeleton.sum()
        sens.append(sen)

        pre = (large_cd * label).sum() / large_cd.sum()
        pres.append(pre)

        # print("能到这，啊")
        num_branch = parsing.max()
        detected_num = 0
        print(f"num_branch{num_branch}")
        for j in range(int(num_branch)):
            branch_label = ((parsing == (j + 1)).astype(np.uint8)) * skeleton
            if (large_cd * branch_label).sum() / branch_label.sum() >= 0.8:
                detected_num += 1
        branch = detected_num / num_branch
        branches.append(branch)

        print(file_list[n * i].split('_')[0], "Length: %0.4f" % (sen), "Precision: %0.4f" % (pre),
              "Branch: %0.4f" % (branch), "Dice: %0.4f" % (curdice), "Se: %0.4f" % (cursensi))

        name =file_list[n * i].split('_')[0]
        logging.info(f'name:{name},length: :{sen:.4f},precision :{pre:.4f},Branch:{branch:.4f},Dice:{curdice:.4f},Se{cursensi:.4f}')
        logging.info(f"---------------------------")
        logging.info(f" 尝试思路如下  ")


    sen1_mean = np.mean(sens)
    sen1_std = np.std(sens)
    pre_mean = np.mean(pres)
    pre_std = np.std(pres)
    branch_mean = np.mean(branches)
    branch_std = np.std(branches)
    dice_mean = np.mean(dice)
    se_mean = np.mean(se)
    dice2_mean = np.mean(dice2)
    jc_mean = np.mean(jc_metric)
    assd_mean = np.mean(assd_metric)
    hd95_mean  = np.mean(hd95_metric)
    fpr_mean = np.mean(FPR)
    tpr_mean = np.mean(TPR)
    logging.info(
        f'sum,length: :{sen:.4f},precision :{pre:.4f},Branch:{branch:.4f},Dice:{curdice:.4f},Se{cursensi:.4f}')
    print("len mean: %0.4f (%0.4f), branch: %0.4f (%0.4f), pre: %0.4f (%0.4f), dice: %0.4f, se: %0.4f" % (
    sen1_mean, sen1_std, branch_mean, branch_std, pre_mean, pre_std, dice_mean, se_mean))
    print(f"dice2:{dice2_mean:.4f},jc_mean:{jc_mean:.4f},assd_mean:{assd_mean:.4f},hd95_mean:{hd95_mean:.4f},fpr_mean:{fpr_mean:.4f},tpr_mean:{tpr_mean:.4f}")

if __name__ == '__main__':
    # data_path = "/home/yyy/PycharmProjects/airway/base_semi/airway_segmentation/preprocessed_data/"
    # save_path = "/home/yyy/PycharmProjects/airway/base_semi/airway_segmentation/results/dataset50_pseudo_label_add_fft_alpha1.0_exp-warmup/test001/"
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # parsing_path = "/home/yyy/PycharmProjects/airway/base_semi/airway_segmentation/tree_parse/"
    # save_dir = '/home/yyy/PycharmProjects/airway/base_semi/airway_segmentation/results/dataset50_pseudo_label_add_fft_alpha1.0_exp-warmup'
    args = parser.parse_args()
    start_time = time.time()
    # logfile = os.path.join(save_dir, 'no_lung_mask_results_log.txt')
    save_path = network_prediction(args)
    evaluation(save_path, parsing_path)
    end_time = time.time()
    print('test finished, time %d seconds' % (end_time - start_time))