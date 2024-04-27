import argparse

parser = argparse.ArgumentParser(description='PyTorch Airway Segmentation')
#parser.add_argument('--model', '-m', metavar='MODEL', default='baseline', help='model')
parser.add_argument('--model', '-m', metavar='MODEL', default='baseline', help='model')

parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
					help='number of data loading workers (default: 4)')#原来8

parser.add_argument('--epochs', default=30, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('--start_val', default=1, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=4, type=int,
					metavar='N', help='mini-batch size (default: 16)')#原来16
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')

parser.add_argument('--lr_sch', type=str, default='exp-warmup', help='lr scheduler')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate [default: 0.001]')
#20
parser.add_argument('--lr_step', type=int, default=15, help='Decay step for learning rate [default: 2500]')

parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for learning rate [default: 0.8]')
parser.add_argument('--min-lr', default=1e-9, type=float, metavar='LR', help='minimum learning rate (default: 1e-4)')
parser.add_argument('--steps', type=int, nargs='+', metavar='N', help='decay steps for multistep scheduler')
#5
parser.add_argument('--rampup_length', type=int, default=10, metavar='EPOCHS', help='length of the ramp-up')
#40
parser.add_argument('--rampdown_length', type=int, default=25, metavar='EPOCHS', help='length of the ramp-down, epochs-rampdown_length begin to ramp-down')

parser.add_argument('--optimizer', type=str, default='adam', help='adam or sgd momentum [default: adam]')
parser.add_argument('--beta1', type=float, default=0.9, help='First decay ratio, for Adam [default: 0.9]')
parser.add_argument('--beta2', type=float, default=0.999, help='Second decay ratio, for Adam [default: 0.999]')
parser.add_argument('--momentum', type=float, default=0.9, help='Gradient descent momentum, for SGD [default: 0.9]')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay coef [default: 0]')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('--resumepart', default=0, type=int, metavar='PARTRESUME',
					help='Resume params. part')
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
					help='directory to save checkpoint (default: none)')
parser.add_argument('--test_dir', default='', type=str, metavar='SAVE',
					help='directory to save checkpoint (default: none)')

parser.add_argument('--test', default=0, type=int, metavar='TEST',
					help='1 do test evaluation, 0 not')
parser.add_argument('--debug', default=0, type=int, metavar='TEST',
					help='debug mode')

parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--MCF', default=1, type=int, metavar='teacher_student',
					help='use MCF 对比算法 ')

parser.add_argument('--deepsupervision', default=0, type=int, metavar='DEEP SUPERVISION',
					help='use deep supervision as auxiliary tasks')
#80,192,304
parser.add_argument('--cubesize', default=[128,128,128], nargs="*", type=int, metavar='cube',
					help='cube size')
# parser.add_argument('--cubesize', default=[80,192,304], nargs="*", type=int, metavar='cube',
# 					help='cube size')
parser.add_argument('--cubesizev', default=None,nargs="*", type=int, metavar='cube',
					help='cube size')
# parser.add_argument('--stridet', default=[48, 80, 80], nargs="*", type=int, metavar='stride',
# 					help='split stride train')

parser.add_argument('--stridet', default=[108,108,108], nargs="*", type=int, metavar='stride',
					help='split stride train')
# parser.add_argument('--stridet', default=[64,96,152], nargs="*", type=int, metavar='stride',
# 					help='split stride train')
#48,80,80
#64,72,72
parser.add_argument('--stridev', default=[108,108,108], nargs="*", type=int, metavar='stride',
					help='split stride val')
# parser.add_argument('--stridev', default=[64,72,72], nargs="*", type=int, metavar='stride',
# 					help='split stride train')
parser.add_argument('--multigpu', default=False, type=bool, metavar='mgpu',
					help='use multiple gpus')
#原来0
parser.add_argument('--gpu', default='0,1', type=str, metavar='mgpu',
					help='use multiple gpus')

parser.add_argument('--flip', default=True, type=bool, metavar='weak augmentation',
					help='if flip or not')
parser.add_argument('--swap', default=False, type=bool, metavar='weak augmentation',
					help='if swap or not')
parser.add_argument('--smooth', default=False, type=bool, metavar='weak augmentation',
					help='if smooth or not')
parser.add_argument('--jitter', default=True, type=bool, metavar='weak augmentation',
					help='if jitter or not')
parser.add_argument('--split_jitter', default=True, type=bool, metavar='weak augmentation',
					help='if split_jitter or not')

parser.add_argument('--colorjitter', default=False, type=bool, help='use color jittering or not')
parser.add_argument('--cut', default=False, type=bool, help='use cutout or not')
parser.add_argument('--cutout', default=48, type=int, help='use to cutout patch')
parser.add_argument('--n_holes', default=1, type=int, help='use to cut patch for n holes')

parser.add_argument('--augment_spatial', default=False, type=bool, help='use augment_spatial or not')
parser.add_argument('--elastic_deform', default=False, type=bool, help='use elastic deform or not')
parser.add_argument('--rotation', default=False, type=bool, help='use rotation or not')
parser.add_argument('--scale', default=False, type=bool, help='use scale or not')
parser.add_argument('--rand_crop', default=False, type=bool, help='use random crop or not')

parser.add_argument('--fft', default=False, type=bool, help='use fft or not')
parser.add_argument('--alpha', default=1.0, type=float, help='AM, FFT WITH MIXUP')
parser.add_argument('--matrix', default=False, type=bool, help='use random matrix or not')

parser.add_argument('--sigma_min', default=4.0, type=float, help='mask')
parser.add_argument('--sigma_max', default=16.0, type=float, help='mask')
parser.add_argument('--p_min', default=0.0, type=float, help='mask')
parser.add_argument('--p_max', default=1.0, type=float, help='mask')

# parser.add_argument('--parsing_path', type=str,
# 					default='/opt/data/private/yyy/base_semi/airway_segmentation/tree_parse/')
# parser.add_argument('--preprocessed_data', type=str,
# 					default='/opt/data/private/yyy/base_semi/airway_segmentation/preprocessed_data')
# parser.add_argument('--parsing_path', type=str,
# 					default='/home/guest/Documents/code/python/semi_airway/airway_segmentation/tree_parse/')
# parser.add_argument('--preprocessed_data', type=str,
# 					default='/home/guest/Documents/code/python/semi_airway/airway_segmentation/preprocessed_data')

#parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=60.0, help='consistency_rampup')

# contrastive loss
parser.add_argument('--contrastive_weight', type=float, default=1.0, help='contrastive_weight')