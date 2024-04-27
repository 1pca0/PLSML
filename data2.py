import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import random
from math import sqrt
from torchvision import transforms
import SimpleITK as sitk
from glob import glob
from scipy.ndimage import gaussian_filter
from scipy.special import erfinv
from utils import save_itk, load_itk_image, lumTrans, load_pickle
from transform import crop, hflip, normalize, resize, blur, cutout
# from batchgenerators.augmentations.color_augmentations import augment_contrast, augment_brightness_additive, \
#     augment_brightness_multiplicative, augment_gamma, augment_illumination, augment_PCA_shift

# from batchgenerators.augmentations.spatial_transformations import augment_spatial

# from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
#     interpolate_img, \
#     rotate_coords_2d, rotate_coords_3d, scale_coords, resize_segmentation, resize_multichannel_image, \
#     elastic_deform_coordinates_2
# from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop as random_crop_aug
# from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_aug
import itertools
from torch.utils.data.sampler import Sampler
import math


def relabel_dataset(dataset):
    unlabeled_idxs = []
    #print(f'dataset_unlabelname{dataset.unlabeled_name}')
    for idx in range(len(dataset.cubelist)):
        # if dataset.paths[idx].endswith('.h5'):
        if dataset.cubelist[idx][0] in dataset.unlabeled_name:
            unlabeled_idxs.append(idx)
    #print(f'unlabel_idx{unlabeled_idxs}')
    labeled_idxs = sorted(set(range(len(dataset))) - set(unlabeled_idxs))
    # unlabeled_idxs = unlabeled_idxs * math.ceil(len(labeled_idxs) / len(unlabeled_idxs))

    labeled_idxs = labeled_idxs * math.ceil(len(unlabeled_idxs)/len(labeled_idxs))
    return labeled_idxs, unlabeled_idxs



class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        # print(len(self.secondary_indices))
        # print(len(self.primary_indices))
        # print(self.primary_batch_size)
        # print(self.secondary_batch_size)
        # assert len(self.primary_indices) >= self.primary_batch_size > 0
        # assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_once(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
	"Collect data into fixed-length chunks or blocks"
	# grouper('ABCDEFG', 3) --> ABC DEF"
	args = [iter(iterable)] * n
	return zip(*args)


class CollateClass(object):
	def __init__(self,  FFT, alpha, matrix):
		self.alpha = alpha
		self.matrix = matrix
		self.FFT = FFT
		pass

	def __call__(self, sample_list):
		bs = len(sample_list) // 2
		x = []
		y = []
		coord = []
		for i in range(len(sample_list)):
			x.append(sample_list[i][0])
			y.append(sample_list[i][1])
			coord.append(sample_list[i][2])
		#  label =
		# x1, x2 = x[:bs]
		# x21, x12 = colorful_spectrum_mix(np.squeeze(x1), np.squeeze(x2), self.alpha, ratio=1.0)
		# x21 = x21[np.newaxis, ...]
		# x12 = x12[np.newaxis, ...]
		#过程中，将输入数据和坐标数据按照不同规则进行组合和变换
		if self.FFT:
			# print('FFT')
			x1 = np.stack((x[:bs]), axis=0)
			y1 = np.stack((y[:]), axis=0)
			coord1 = np.stack((coord[:bs]), axis=0)
			img1, img2 = x[bs:]

			img21, img12 = colorful_spectrum_mix(np.squeeze(img1), np.squeeze(img2), self.alpha, ratio=1.0, matrix=self.matrix)
			img21 = img21[np.newaxis, ...]
			img12 = img12[np.newaxis, ...]


			# img211 = img211[np.newaxis, ...]
			# img122 = img122[np.newaxis, ...]
			#组合频谱图像：img21 和 img12 是两个图像的频谱（傅里叶变换后的表示）。
			# x2 将这两个频谱在新的第一个维度上进行堆叠，形成一个包含两个频谱的张量
			x2 = np.stack((img21, img12), axis=0)
			x3 = np.stack((img1, img2), axis=0)

			new_x = np.concatenate((x1, x2, x3), axis=0)

			coord2 = np.stack((coord[bs:]), axis=0)
			new_coord = np.concatenate((coord1, coord2, coord2), axis=0)
			new_x = torch.from_numpy(new_x).float()
			new_y = np.concatenate((y1, y[bs:]), axis=0)
			new_y = torch.from_numpy(new_y).float()
			new_coord = torch.from_numpy(new_coord).float()
		else:
			new_x = np.stack((x[:]), axis=0)
			new_coord = np.stack((coord[:]), axis=0)
			y1 = np.stack((y[:]), axis=0)
			new_x = torch.from_numpy(new_x).float()
			new_y = torch.from_numpy(y1).float()
			new_coord = torch.from_numpy(new_coord).float()

		pass

		return new_x, new_y, new_coord

# def collate_fn(sample_list):
# 	bs = len(sample_list)// 2
# 	x=[]
# 	y=[]
# 	coord =[]
# 	for i in range(len(sample_list)):
# 		x.append(sample_list[i][0])
# 		y.append(sample_list[i][1])
# 		coord.append(sample_list[i][2])
# 	# label =
# 	x1 = torch.stack((x[:bs]), dim=0)
# 	y1 = torch.stack((y[:bs]), dim=0)
# 	coord1 = torch.stack((coord[:bs]), dim=0)
# 	x2 = torch.cat((x[bs:]), dim=0)
# 	y2 = torch.cat((y[bs:]), dim=0)
# 	coord2 = torch.cat((coord[bs:]), dim=0)
# 	new_x = torch.cat((x1,x2), dim=0)
# 	new_y = torch.cat((y1, y2), dim=0)
# 	new_coord = torch.cat((coord1, coord2), dim=0)
# 	pass
# 	return new_x,new_y, new_coord

class AirwayData(Dataset):
	"""
	Generate dataloader
	"""
	def __init__(self, config, args, split_comber=None, debug=False):
		"""
		:param config: configuration from model
		:param phase: training or validation or testing
		:param split_comber: split-combination-er
		:param debug: debug mode to check few data
		:param random_select: use partly, randomly chosen data for training
		"""
		self.augtype = config['augtype']
		self.split_comber = split_comber
		self.debug_flag = debug

		"""
		specify the path and data split
		"""
		self.datapath = config['dataset_path']
		self.dataset = load_pickle(config['dataset_split'])

class train_dataset(AirwayData):

	def __init__(self, config, args, split_comber=None, debug=False):
		super(train_dataset, self).__init__(config, args, split_comber=split_comber, debug=debug)

		print("-------------------------Load all train data into memory---------------------------")

		"""
		count the number of cases
		"""
		# labellist = []
		cubelist = []
		self.caseNumber = 0
		allimgdata_memory = {}
		alllabeldata_memory = {}

		file_names = self.dataset['train']
		#可以去掉file_names['exact09']
		data_file_names = file_names['lidc'] + file_names['exact09']
		file_num = len(data_file_names)
		if self.debug_flag:
			data_file_names = file_names['lidc'][:1] + file_names['exact09'][:1]
			file_num = len(data_file_names)
		self.caseNumber += file_num

		print("total %s case number: %d" % ('train', self.caseNumber))

		for raw_path in data_file_names:

			raw_path = os.path.join(self.datapath, raw_path.split('/')[-1] + '.gz')
			assert (os.path.exists(raw_path) is True)
			label_path = raw_path.replace('clean_hu', 'label')
			assert (os.path.exists(label_path) is True)

			imgs, origin, spacing = load_itk_image(raw_path)
			splits, nzhw, orgshape = self.split_comber.split_id(imgs)
			data_name = raw_path.split('/')[-1].split('_clean_hu')[0]

			print("Name: %s, # of splits: %d" % (data_name, len(splits)))
			labels, _, _ = load_itk_image(label_path)

			allimgdata_memory[data_name] = [imgs, origin, spacing]
			alllabeldata_memory[data_name] = labels
			cube_train = []

			for j in range(len(splits)):
				"""
				check if this sub-volume cube is suitable
				"""
				cursplit = splits[j]
				labelcube = labels[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1],
							cursplit[2][0]:cursplit[2][1]]
				curnumlabel = np.sum(labelcube)
				# labellist.append(curnumlabel)
				if curnumlabel > 0:  # filter out those zero-0 labels
					curlist = [data_name, cursplit, j, nzhw, orgshape, 'Y']
					cube_train.append(curlist)

			random.shuffle(cube_train)
			cubelist += cube_train

		self.allimgdata_memory = allimgdata_memory
		self.alllabeldata_memory = alllabeldata_memory

		# if self.rand_sel and self.phase == 'train':
		# 	assert (len(cubelist) == self.caseNumber)
		# 	mean_labelnum = np.mean(np.array(labellist))
		# 	print('mean label number: %d' % (mean_labelnum))
		# 	print('total patches: ', self.patch_per_case * self.caseNumber)

		random.shuffle(cubelist)
		self.cubelist = cubelist

		print('---------------------Initialization Done---------------------')
		print('Phase: %s total cubelist number: %d' % ('train', len(self.cubelist)))
		print()

	def __len__(self):
		return len(self.cubelist)

	def __getitem__(self, idx):
		"""
		:param idx: index of the batch
		:return: wrapped data tensor
		"""
		t = time.time()
		np.random.seed(int(str(t % 1)[2:7]))  # seed according to time
		curlist = self.cubelist[idx]

		# train: [data_name, cursplit, j, nzhw, orgshape, 'Y']
		# val/test: [data_name, cursplit, j, nzhw, orgshape, 'N']

		curNameID = curlist[0]
		cursplit = curlist[1]
		curShapeOrg = curlist[4]
		curtransFlag = curlist[5]

		if curtransFlag == 'Y' and self.augtype['split_jitter'] is True:
			# random jittering during the training
			cursplit = augment_split_jittering(cursplit, curShapeOrg)

		####################################################################
		imginfo = self.allimgdata_memory[curNameID]
		imgs, origin, spacing = imginfo[0], imginfo[1], imginfo[2]
		curcube = imgs[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
		curcube = (curcube.astype(np.float32)) / 255.0

		####################################################################

		# calculate the coordinate for coordinate-aware convolution
		start = [float(cursplit[0][0]), float(cursplit[1][0]), float(cursplit[2][0])]
		normstart = ((np.array(start).astype('float') / np.array(curShapeOrg).astype('float')) - 0.5) * 2.0
		crop_size = [curcube.shape[0], curcube.shape[1], curcube.shape[2]]
		stride = 1.0
		normsize = (np.array(crop_size).astype('float') / np.array(curShapeOrg).astype('float')) * 2.0
		xx, yy, zz = np.meshgrid(np.linspace(normstart[0], normstart[0] + normsize[0], int(crop_size[0])),
								 np.linspace(normstart[1], normstart[1] + normsize[1], int(crop_size[1])),
								 np.linspace(normstart[2], normstart[2] + normsize[2], int(crop_size[2])),
								 indexing='ij')
		coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, ...]], 0).astype('float')
		assert (coord.shape[0] == 3)

		label = self.alllabeldata_memory[curNameID]
		label = label[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
		label = (label > 0)
		label = label.astype('float')
		####################################################################

		########################Data augmentation##############################
		if curtransFlag == 'Y':
			curcube, label, coord = augment(curcube, label, coord,
											ifflip=self.augtype['flip'], ifswap=self.augtype['swap'],
											ifsmooth=self.augtype['smooth'], ifjitter=self.augtype['jitter'])
		curcube = curcube[np.newaxis, ...]
		label = label[np.newaxis, ...]
		x = torch.from_numpy(curcube).float()
		y = torch.from_numpy(label).float()
		coord = torch.from_numpy(coord).float()

		return x,y,coord

class semi_train_dataset(AirwayData):

	def __init__(self, config, args, split_comber=None, debug=False):
		super(semi_train_dataset, self).__init__(config, args, split_comber=split_comber, debug=debug)

		self.augment_spatial = args.augment_spatial
		self.RD = args.elastic_deform
		self.rotation = args.rotation
		self.scale = args.scale
		self.rand_crop = args.rand_crop
		self.colorjitter = args.colorjitter
		self.cutout = args.cut
		self.n = args.n_holes
		self.crop = args.cutout
		self.FFT = args.fft
		self.alpha = args.alpha
		self.sigma_min = args.sigma_min
		self.sigma_max = args.sigma_max
		self.p_min = args.p_min
		self.p_max = args.p_max
		self.unlabeled_name = []
		#test当做无标签数据
		#train里面包含了test,进行data_file_names时对无标签和有标签同时处理
		unlabel_data_file_names = self.dataset['train']['unlabel']
		for i in unlabel_data_file_names:
			i = os.path.join(self.datapath, i.split('/')[-1] + '.gz')#LIDC-IDRI-0032_clean_hu.nii
			#assert (os.path.exists(i) is True)
			label_path = i.replace('clean_hu', 'label')
			#assert (os.path.exists(label_path) is True)
			#self.unlabeled_name['preprocessed_data\\LIDC-IDRI-0333', 'preprocessed_data\\LIDC-IDRI-0308',
			self.unlabeled_name.append(i.split('\\')[-1].split('_clean_hu')[0])#LIDC-IDRI-0032
		print(f'self.unlabeled_name{self.unlabeled_name}')
		print("-------------------------Load all semi train data into memory---------------------------")

		"""
		count the number of cases
		"""
		cubelist = []
		self.caseNumber = 0
		allimgdata_memory = {}
		alllabeldata_memory = {}
		file_names = self.dataset['train']
		data_file_names = file_names['label']
		file_num = len(data_file_names)+len(unlabel_data_file_names)
		if self.debug_flag:
			#data_file_names = file_names['lidc'][:4] + file_names['exact09'][:1]
			data_file_names = file_names['label']
			file_num = len(data_file_names)
		self.caseNumber += file_num

		print("total %s case number: %d"%('semi_train', self.caseNumber))

		for raw_path in data_file_names:
			name = raw_path.split('/')[-1].split('_clean_hu')[0]
			print(f'name{name}')
			raw_path = os.path.join(self.datapath, raw_path.split('/')[-1] + '.gz')
			assert(os.path.exists(raw_path) is True)
			label_path = raw_path.replace('clean_hu', 'label')
			assert (os.path.exists(label_path) is True)
			#imgs是加载的标签图像数据，origin是图像的原点坐标，spacing是图像的像素间距。
			imgs, origin, spacing = load_itk_image(raw_path)
			#nzhw = [nz, nh, nw],orgshape = [z, h, w]原始形状
			# idcs = [[sz, ez],[sh, eh], [sw, ew],idx]
			# splits.append(idcs)划分的大小
			splits, nzhw, orgshape = self.split_comber.split_id(imgs)

			data_name = raw_path.split('\\')[-1].split('_clean_hu')[0]
			# print(f'data_name {data_name}')
			print("Name: %s, # of splits: %d"%(data_name, len(splits)))
			labels, _, _ = load_itk_image(label_path)

			allimgdata_memory[data_name] = [imgs, origin, spacing]
			alllabeldata_memory[data_name] = labels
			# numsq = 0
			# if name in self.unlabeled_name:
			# 	numsq+=1
			# 	for j in range(len(splits)):
			# 		"""
			# 		check if this cube is suitable
			# 		"""
			# 		cursplit = splits[j]
			# 		curlist = [data_name, cursplit, j, nzhw, orgshape, 'Y']
			# 		cubelist.append(curlist)

			cube_train = []

			for j in range(len(splits)):
				"""
				check if this sub-volume cube is suitable
				"""
				cursplit = splits[j]
				labelcube = labels[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
				curnumlabel = np.sum(labelcube)
				if curnumlabel > 0:  # filter out those zero-0 labels
					curlist = [data_name, cursplit, j, nzhw, orgshape, 'Y']
					cube_train.append(curlist)

			random.shuffle(cube_train)
			cubelist += cube_train
			# print(numsq)
		for unlabel in unlabel_data_file_names:
			name = unlabel.split('/')[-1].split('_clean_hu')[0]
			print(f'name{name}')
			raw_path = os.path.join(self.datapath, unlabel.split('/')[-1] + '.gz')
			assert (os.path.exists(raw_path) is True)
			label_path = raw_path.replace('clean_hu', 'label')
			assert (os.path.exists(label_path) is True)
			# imgs是加载的标签图像数据，origin是图像的原点坐标，spacing是图像的像素间距。
			imgs, origin, spacing = load_itk_image(raw_path)
			# nzhw = [nz, nh, nw],orgshape = [z, h, w]原始形状
			# idcs = [[sz, ez],[sh, eh], [sw, ew],idx]
			# splits.append(idcs)划分的大小
			splits, nzhw, orgshape = self.split_comber.split_id(imgs)
			data_name = unlabel.split('/')[-1].split('_clean_hu')[0]

			print("Name: %s, # of splits: %d" % (data_name, len(splits)))
			labels, _, _ = load_itk_image(label_path)

			allimgdata_memory[data_name] = [imgs, origin, spacing]
			alllabeldata_memory[data_name] = labels
			for j in range(len(splits)):
				"""
				check if this cube is suitable
				"""
				cursplit = splits[j]
				curlist = [data_name, cursplit, j, nzhw, orgshape, 'Y']
				cubelist.append(curlist)
		self.allimgdata_memory = allimgdata_memory
		self.alllabeldata_memory = alllabeldata_memory

		random.shuffle(cubelist)
		self.cubelist = cubelist

		print('---------------------Initialization Done---------------------')
		print('Phase: %s total cubelist number: %d' % ('semi_train', len(self.cubelist)))
		print()

	def __len__(self):
		return len(self.cubelist)

	def __getitem__(self, idx):
		"""
		:param idx: index of the batch
		:return: wrapped data tensor and name, shape, origin, etc.
		"""
		t = time.time()
		np.random.seed(int(str(t % 1)[2:7]))  # seed according to time
		curlist = self.cubelist[idx]

		# train: [data_name, cursplit, j, nzhw, orgshape, 'Y']
		# val/test: [data_name, cursplit, j, nzhw, orgshape, 'N']

		curNameID = curlist[0]
		cursplit = curlist[1]
		# curSplitID = curlist[2]
		# curnzhw = curlist[3]
		curShapeOrg = curlist[4]
		curtransFlag = curlist[5]
		#数据增强
		if curtransFlag == 'Y' and self.augtype['split_jitter'] is True:
			# random jittering during the training
			cursplit = augment_split_jittering(cursplit, curShapeOrg)

		####################################################################
		imginfo = self.allimgdata_memory[curNameID]
		imgs, origin, spacing = imginfo[0], imginfo[1], imginfo[2]
		curcube = imgs[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
		curcube = (curcube.astype(np.float32)) / 255.0

		####################################################################

		# calculate the coordinate for coordinate-aware convolution
		start = [float(cursplit[0][0]), float(cursplit[1][0]), float(cursplit[2][0])]
		normstart = ((np.array(start).astype('float') / np.array(curShapeOrg).astype('float')) - 0.5) * 2.0
		crop_size = [curcube.shape[0], curcube.shape[1], curcube.shape[2]]
		stride = 1.0
		normsize = (np.array(crop_size).astype('float') / np.array(curShapeOrg).astype('float')) * 2.0
		xx, yy, zz = np.meshgrid(np.linspace(normstart[0], normstart[0] + normsize[0], int(crop_size[0])),
								 np.linspace(normstart[1], normstart[1] + normsize[1], int(crop_size[1])),
								 np.linspace(normstart[2], normstart[2] + normsize[2], int(crop_size[2])),
								 indexing='ij')
		coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, ...]], 0).astype('float')
		assert (coord.shape[0] == 3)

		label = self.alllabeldata_memory[curNameID]
		label = label[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
		label = (label > 0)
		label = label.astype('float')
		####################################################################

		########################Weaak Data augmentation##############################
		#数据增强
		if curtransFlag == 'Y':
			curcube, label, coord = augment(curcube, label, coord,
											ifflip=self.augtype['flip'], ifswap=self.augtype['swap'],
											ifsmooth=self.augtype['smooth'], ifjitter=self.augtype['jitter'])
		# flag = True
		# curcube2,label,coord=augment(curcube,label,coord,ifflip=True,ifswap=True,ifsmooth = True,ifjitter=True)

		# ########################Strong Data augmentation##############################
		# if curNameID in self.unlabeled_name:
		# 	# img = transforms.RandomGrayscale(p=0.2)(curcube)
		#
		# 	if self.FFT:
		# 		img1 = curcube
		# 		img_idx = random.randint(0, len(self.cubelist) - 1)
		# 		curlist = self.cubelist[img_idx]
		# 		while curNameID == curlist[0]:
		# 			img_idx = random.randint(0, len(self.cubelist) - 1)
		# 			curlist = self.cubelist[img_idx]
		# 		cursplit = curlist[1]
		# 		imginfo = self.allimgdata_memory[curlist[0]]
		# 		imgs = imginfo[0]
		# 		curcube = imgs[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1],
		# 				  cursplit[2][0]:cursplit[2][1]]
		# 		img2 = (curcube.astype(np.float32)) / 255.0
		# 		# curcube = colorful_spectrum_mix(img1, img2, self.sigma_min, self.sigma_max, self.p_min, self.p_max,
		# 		# 								ratio=1.0)
		# 		img21, img12 = colorful_spectrum_mix(img1, img2, self.alpha, ratio=1.0)
		# 		img21 = img21[np.newaxis, ...]
		# 		img12 = img12[np.newaxis, ...]
		#
		# 		curcube  = np.stack((img21, img12), axis=0)
		#
		# 		# label
		# 		label2 = self.alllabeldata_memory[curlist[0]]
		# 		label2 = label2[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1],
		# 				cursplit[2][0]:cursplit[2][1]]
		# 		label2 = (label2 > 0)
		# 		label2 = label2.astype('float')
		# 		label = label[np.newaxis, ...]
		# 		label2 = label2[np.newaxis, ...]
		#
		# 		label = np.stack((label, label2), axis=0)
		#
		# 		# coord
		# 		start = [float(cursplit[0][0]), float(cursplit[1][0]), float(cursplit[2][0])]
		# 		curShapeOrg =curlist[4]
		# 		normstart = ((np.array(start).astype('float') / np.array(curShapeOrg).astype('float')) - 0.5) * 2.0
		# 		crop_size = [img1.shape[0], img1.shape[1], img1.shape[2]]
		# 		normsize = (np.array(crop_size).astype('float') / np.array(curShapeOrg).astype('float')) * 2.0
		# 		xx, yy, zz = np.meshgrid(np.linspace(normstart[0], normstart[0] + normsize[0], int(crop_size[0])),
		# 								 np.linspace(normstart[1], normstart[1] + normsize[1], int(crop_size[1])),
		# 								 np.linspace(normstart[2], normstart[2] + normsize[2], int(crop_size[2])),
		# 								 indexing='ij')
		# 		coord2 = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, ...]], 0).astype(
		# 			'float')
		# 		assert (coord2.shape[0] == 3)
		# 		coord = np.stack((coord, coord2), axis=0)
		#
		# 	# if self.colorjitter:
		# 	# 	curcube = augment_contrast(curcube, contrast_range=(0.8, 1.2), preserve_range=True, per_channel=True,
		# 	# 							   p_per_channel=1)
		# 	# 	curcube = augment_brightness_additive(curcube, mu=0, sigma=0.1)
		# 	# # curcube = augment_brightness_multiplicative(curcube, multiplier_range=(0.5, 1.5))
		# 	# if self.cutout:
		# 	# 	curcube, label = cutout(curcube, label, p=0.5, length=self.crop, n_holes=self.n)
		# 	# if self.augment_spatial:
		# 	# 	curcube, label = augment_spatial(curcube, label, curcube.shape, do_elastic_deform=self.RD,
		# 	# 									 do_rotation=self.rotation, do_scale=self.scale,
		# 	# 									 random_crop=self.rand_crop)
		#
		# 	x = torch.from_numpy(curcube).float()
		# 	y = torch.from_numpy(label).float()
		# 	coord = torch.from_numpy(coord).float()
		#
		# else:
		# 	curcube = curcube[np.newaxis, ...]
		# 	label = label[np.newaxis, ...]
		# 	x = torch.from_numpy(curcube).float()
		# 	y = torch.from_numpy(label).float()
		# 	coord = torch.from_numpy(coord).float()
		#要加不同的扰动，复制x,y为想x1,x2,y1,y2
			x = curcube[np.newaxis, ...]
			y = label[np.newaxis, ...]

			return x, y, coord

class val_dataset(AirwayData):

	def __init__(self, config, args, split_comber=None, debug=False):
		super(val_dataset, self).__init__(config, args, split_comber=split_comber, debug=debug)

		print("-------------------------Load all val data into memory---------------------------")

		"""
		count the number of cases
		"""
		cubelist = []
		self.caseNumber = 0
		allimgdata_memory = {}
		alllabeldata_memory = {}
		file_names = self.dataset['val']
		#data_file_names = file_names['lidc'] + file_names['exact09']
		data_file_names = file_names
		file_num = len(data_file_names)
		if self.debug_flag:
			data_file_names = file_names['lidc'][:] + file_names['exact09'][:1]
			file_num = len(data_file_names)

		self.caseNumber += file_num
		print("total %s case number: %d" % ('val', self.caseNumber))

		for raw_path in data_file_names:
			raw_path = os.path.join(self.datapath, raw_path.split('/')[-1] + '.gz')
			assert (os.path.exists(raw_path) is True)

			label_path = raw_path.replace('clean_hu', 'label')
			assert (os.path.exists(label_path) is True)

			imgs, origin, spacing = load_itk_image(raw_path)
			splits, nzhw, orgshape = self.split_comber.split_id(imgs)
			data_name = raw_path.split('/')[-1].split('_clean_hu')[0]
			print("Name: %s, # of splits: %d" % (data_name, len(splits)))
			labels, _, _ = load_itk_image(label_path)

			allimgdata_memory[data_name] = [imgs, origin, spacing]
			alllabeldata_memory[data_name] = labels

			for j in range(len(splits)):
				cursplit = splits[j]
				curlist = [data_name, cursplit, j, nzhw, orgshape, 'N']
				cubelist.append(curlist)

		self.allimgdata_memory = allimgdata_memory
		self.alllabeldata_memory = alllabeldata_memory
		self.cubelist = cubelist

		print('---------------------Initialization Done---------------------')
		print('Phase: %s total cubelist number: %d' % ('val', len(self.cubelist)))
		print()


	def __len__(self):
		return len(self.cubelist)

	def __getitem__(self, idx):

		"""
		:param idx: index of the batch
		:return: wrapped data tensor and name, shape, origin, etc.
		"""
		t = time.time()
		np.random.seed(int(str(t % 1)[2:7]))  # seed according to time
		curlist = self.cubelist[idx]

		# train: [data_name, cursplit, j, nzhw, orgshape, 'Y']
		# val/test: [data_name, cursplit, j, nzhw, orgshape, 'N']

		curNameID = curlist[0]
		cursplit = curlist[1]
		curSplitID = curlist[2]
		curnzhw = curlist[3]
		curShapeOrg = curlist[4]
		# curtransFlag = curlist[5]

		####################################################################
		imginfo = self.allimgdata_memory[curNameID]
		imgs, origin, spacing = imginfo[0], imginfo[1], imginfo[2]
		curcube = imgs[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
		curcube = (curcube.astype(np.float32)) / 255.0

		####################################################################

		# calculate the coordinate for coordinate-aware convolution
		# 为坐标感知卷积计算坐标
		#start 变量包含了当前切块（subcube）在原始数据中的起始位置，是一个三维坐标。
		start = [float(cursplit[0][0]), float(cursplit[1][0]), float(cursplit[2][0])]

		normstart = ((np.array(start).astype('float') / np.array(curShapeOrg).astype('float')) - 0.5) * 2.0
		crop_size = [curcube.shape[0], curcube.shape[1], curcube.shape[2]]
		stride = 1.0
		#normsize 对 crop_size 进行了归一化，同样映射到[-1, 1]的范围内
		normsize = (np.array(crop_size).astype('float') / np.array(curShapeOrg).astype('float')) * 2.0
		#xx, yy, zz 使用 np.meshgrid 函数生成了一个三维坐标网格，表示了当前切块在归一化空间中的位置
		xx, yy, zz = np.meshgrid(np.linspace(normstart[0], normstart[0] + normsize[0], int(crop_size[0])),
								 np.linspace(normstart[1], normstart[1] + normsize[1], int(crop_size[1])),
								 np.linspace(normstart[2], normstart[2] + normsize[2], int(crop_size[2])),
								 indexing='ij')
		#coord 变量将 xx, yy, zz 三个坐标合并成一个三维数组，表示了当前切块内每个体素的三维坐标
		coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, ...]], 0).astype('float')
		assert (coord.shape[0] == 3)

		label = self.alllabeldata_memory[curNameID]
		#label 是标签数据，表示医学影像中的结构标签。在这里，根据 cursplit 对 label 进行了裁剪操作，并将非零值设置为1，以二值化标签。
		label = label[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
		label = (label > 0)
		label = label.astype('float')
		####################################################################
		# curSplitID = [curSplitID]

		curnzhw = np.array(curnzhw)
		curShapeOrg = np.array(curShapeOrg)
		#######################################################################

		# curNameID = curNameID
		curcube = curcube[np.newaxis, ...]
		label = label[np.newaxis, ...]

		return torch.from_numpy(curcube).float(), torch.from_numpy(label).float(), \
			   torch.from_numpy(coord).float(), torch.from_numpy(origin), \
			   torch.from_numpy(spacing), curNameID, curSplitID, \
			   torch.from_numpy(curnzhw), torch.from_numpy(curShapeOrg)

class test_dataset(AirwayData):

	def __init__(self, config, args, split_comber=None, debug=False):
		super(test_dataset, self).__init__(config, args, split_comber=split_comber, debug=debug)

		print("-------------------------Load all test data into memory---------------------------")

		"""
		count the number of cases
		"""
		cubelist = []
		self.caseNumber = 0
		allimgdata_memory = {}
		alllabeldata_memory = {}
		file_names = self.dataset['test']
		#data_file_names = file_names['lidc'] + file_names['exact09']
		data_file_names = file_names
		file_num = len(data_file_names)
		if self.debug_flag:
			data_file_names = file_names['lidc'][:1] + file_names['exact09'][:1]
			file_num = len(data_file_names)

		self.caseNumber += file_num
		print("total %s case number: %d" % ('test', self.caseNumber))

		for raw_path in data_file_names:
			raw_path = os.path.join(self.datapath, raw_path.split('/')[-1] + '.gz')
			assert (os.path.exists(raw_path) is True)

			label_path = raw_path.replace('clean_hu', 'label')
			assert (os.path.exists(label_path) is True)

			imgs, origin, spacing = load_itk_image(raw_path)
			splits, nzhw, orgshape = self.split_comber.split_id(imgs)
			data_name = raw_path.split('/')[-1].split('_clean_hu')[0]
			print("Name: %s, # of splits: %d" % (data_name, len(splits)))
			labels, _, _ = load_itk_image(label_path)

			allimgdata_memory[data_name] = [imgs, origin, spacing]
			alllabeldata_memory[data_name] = labels

			for j in range(len(splits)):
				"""
				check if this cube is suitable
				"""
				cursplit = splits[j]
				curlist = [data_name, cursplit, j, nzhw, orgshape, 'N']
				cubelist.append(curlist)

		self.allimgdata_memory = allimgdata_memory
		self.alllabeldata_memory = alllabeldata_memory

		self.cubelist = cubelist

		print('---------------------Initialization Done---------------------')
		print('Phase: %s total cubelist number: %d' % ('test', len(self.cubelist)))
		print()

	def __len__(self):
		return len(self.cubelist)

	def __getitem__(self, idx):

		"""
		:param idx: index of the batch
		:return: wrapped data tensor and name, shape, origin, etc.
		"""
		t = time.time()
		np.random.seed(int(str(t % 1)[2:7]))  # seed according to time
		curlist = self.cubelist[idx]

		# train: [data_name, cursplit, j, nzhw, orgshape, 'Y']
		# val/test: [data_name, cursplit, j, nzhw, orgshape, 'N']

		curNameID = curlist[0]
		cursplit = curlist[1]
		curSplitID = curlist[2]
		curnzhw = curlist[3]
		curShapeOrg = curlist[4]
		# curtransFlag = curlist[5]

		####################################################################
		imginfo = self.allimgdata_memory[curNameID]
		imgs, origin, spacing = imginfo[0], imginfo[1], imginfo[2]
		curcube = imgs[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
		curcube = (curcube.astype(np.float32)) / 255.0

		####################################################################

		# calculate the coordinate for coordinate-aware convolution
		start = [float(cursplit[0][0]), float(cursplit[1][0]), float(cursplit[2][0])]
		normstart = ((np.array(start).astype('float') / np.array(curShapeOrg).astype('float')) - 0.5) * 2.0
		crop_size = [curcube.shape[0], curcube.shape[1], curcube.shape[2]]
		stride = 1.0
		normsize = (np.array(crop_size).astype('float') / np.array(curShapeOrg).astype('float')) * 2.0
		xx, yy, zz = np.meshgrid(np.linspace(normstart[0], normstart[0] + normsize[0], int(crop_size[0])),
								 np.linspace(normstart[1], normstart[1] + normsize[1], int(crop_size[1])),
								 np.linspace(normstart[2], normstart[2] + normsize[2], int(crop_size[2])),
								 indexing='ij')
		coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, ...]], 0).astype('float')
		assert (coord.shape[0] == 3)

		label = self.alllabeldata_memory[curNameID]
		label = label[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
		label = (label > 0)
		label = label.astype('float')
		####################################################################

		#curSplitID = [curSplitID]

		curnzhw = np.array(curnzhw)
		curShapeOrg = np.array(curShapeOrg)
		#######################################################################

		#curNameID = [curNameID]


		curcube = curcube[np.newaxis, ...]
		label = label[np.newaxis, ...]

		return torch.from_numpy(curcube).float(), torch.from_numpy(label).float(), \
			   torch.from_numpy(coord).float(), torch.from_numpy(origin), \
			   torch.from_numpy(spacing), curNameID, curSplitID, \
			   torch.from_numpy(curnzhw), torch.from_numpy(curShapeOrg)

def generate_mixing_mask(img_size, sigma_min, sigma_max, p_min, p_max):
	sigma = np.exp(np.random.uniform(np.log(sigma_min), np.log(sigma_max)))
	p = np.random.uniform(p_min, p_max)
	N = np.random.normal(size= img_size)
	noise_smooth = gaussian_filter(N, sigma)
	t = erfinv(p*2 -1)*(2**0.5)*noise_smooth.std() +noise_smooth.mean()
	return (noise_smooth>t).astype(float)

def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0, matrix=False):
	"""Input image size: ndarray of [H, W, C]"""
	#参数检查和裁剪： 确保输入的两个图像形状相同，
	# 然后计算裁剪的区域大小，以及裁剪的起始位置。裁剪区域大小是原始图像大小的平方根乘以 ratio。
	assert img1.shape == img2.shape
	h, w, c = img1.shape
	# mask = generate_mixing_mask((h, w, c), sigma_min, sigma_max, p_min, p_max)
	h_crop = int(h * sqrt(ratio))
	w_crop = int(w * sqrt(ratio))
	c_crop = int(c * sqrt(ratio))
	h_start = h // 2 - h_crop // 2
	w_start = w // 2 - w_crop // 2
	c_start = c // 2 - c_crop // 2
	#傅里叶变换： 对输入的两个图像进行三维傅里叶变换，并分别得到幅度和相位。
	#np.fft.fftn 用于计算多维离散傅里叶变换，而 np.fft.fftshift 用于将频谱移到中心位置。
	img1_fft = np.fft.fftn(img1, axes=(0, 1, 2))
	img2_fft = np.fft.fftn(img2, axes=(0, 1, 2))
	img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
	img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

	img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1, 2))
	img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1, 2))

	if matrix:
		#生成一个与 img1_abs 相同形状的随机矩阵 lam1，其中元素的值在 [0, alpha) 范围内均匀分布
		lam1 = np.random.uniform(0, alpha, img1_abs.shape)
		#对频谱进行混合： 将 img1_abs 和 img2_abs 分别与 lam1 相乘，实现了频谱混合。这一步的目的是引入随机性，使得混合的程度随机。
		img1_abs *= lam1
		img2_abs *= lam1
		#再次生成随机混合矩阵lam2： 生成一个在 [0, alpha) 范围内均匀分布的随机数 lam2。
		lam2 = np.random.uniform(0, alpha)
		#在裁剪区域进行频谱混合： 将 img1_abs 和 img2_abs 在裁剪的区域内进行混合，混合的权重由 lam2 决定。
		img1_abs_ = np.copy(img1_abs)
		img2_abs_ = np.copy(img2_abs)
		img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop, c_start:c_start + c_crop] = \
			lam2 * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop, c_start:c_start + c_crop] + (1 - lam2) * img1_abs_[
								   h_start:h_start + h_crop,
								   w_start:w_start + w_crop,
								   c_start:c_start + c_crop]

		img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop, c_start:c_start + c_crop] = \
			lam2 * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop, c_start:c_start + c_crop] + (1 - lam2) * img2_abs_[
																							  h_start:h_start + h_crop,
																							  w_start:w_start + w_crop,
																							  c_start:c_start + c_crop]

	else:
		lam1 = np.random.uniform(0, alpha)

		img1_abs_ = np.copy(img1_abs)
		img2_abs_ = np.copy(img2_abs)
		img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop, c_start:c_start + c_crop] = \
			lam1 * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop, c_start:c_start + c_crop] + (1 - lam1) * img1_abs_[
								   h_start:h_start + h_crop,
								   w_start:w_start + w_crop,
								   c_start:c_start + c_crop]

		img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop, c_start:c_start + c_crop] = \
			lam1 * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop, c_start:c_start + c_crop] + (1 - lam1) * img2_abs_[
																							  h_start:h_start + h_crop,
																							  w_start:w_start + w_crop,
																							  c_start:c_start + c_crop]
	#执行了频域到时域的逆变换
	#逆移频操作： np.fft.ifftshift 用于将频谱进行逆移，将零频率移到频谱中心。这是逆变换前的准备步骤。
	img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1, 2))
	img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1, 2))

	img21 = img1_abs * (np.e ** (1j * img1_pha))
	img12 = img2_abs * (np.e ** (1j * img2_pha))
	#频域到时域逆变换： 利用逆傅立叶变换 np.fft.ifftn 将频域信号转换为时域信号。img21 和 img12 是逆变换后的实部。
	img21 = np.real(np.fft.ifftn(img21, axes=(0, 1, 2)))
	img12 = np.real(np.fft.ifftn(img12, axes=(0, 1, 2)))
	#np.clip 将逆变换后的像素值限制在 [0, 1] 的范围内，确保图像的像素值在有效范围内。
	img21 = np.clip(img21, 0., 1.)
	img12 = np.clip(img12, 0., 1.)

	# img11_abs = np.copy(img1_abs)
	# img22_abs = np.copy(img2_abs)

	# lam2 = np.random.uniform(0, alpha, img11_abs.shape)
	# img11_abs[h_start:h_start + h_crop, w_start:w_start + w_crop, c_start:c_start + c_crop] = \
	# 	lam2 * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop, c_start:c_start + c_crop] + (
	# 				1 - lam2) * img1_abs_[
	# 						   h_start:h_start + h_crop,
	# 						   w_start:w_start + w_crop,
	# 						   c_start:c_start + c_crop]
	#
	# img22_abs[h_start:h_start + h_crop, w_start:w_start + w_crop, c_start:c_start + c_crop] = \
	#     lam2 * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop, c_start:c_start + c_crop] + (1 - lam2) * img2_abs_[
	#                                                                                       h_start:h_start + h_crop,
	#                                                                                       w_start:w_start + w_crop,
	#                                                                                       c_start:c_start + c_crop]
	#
	# img211 = img11_abs * (np.e ** (1j * img1_pha))
	# img122 = img22_abs * (np.e ** (1j * img2_pha))
	# img211 = np.real(np.fft.ifftn(img211, axes=(0, 1, 2)))
	# img122 = np.real(np.fft.ifftn(img122, axes=(0, 1, 2)))
	# img211 = np.clip(img211, 0., 1.)
	# img122 = np.clip(img122, 0., 1.)

	# return img21, img12, img211, img122
	return img21, img12

# def augment_spatial(data, seg, patch_size, patch_center_dist_from_border=64,
#                     do_elastic_deform=True, alpha=(0., 1000.), sigma=(5., 5.),
#                     do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
#                     do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
#                     border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, p_el_per_sample=1,
#                     p_scale_per_sample=1, p_rot_per_sample=1, independent_scale_for_each_axis=False,
#                     p_rot_per_axis: float = 1, p_independent_scale_per_axis: int = 1):
# 	dim = len(patch_size)
# 	seg_result = None
# 	if seg is not None:
# 		if dim == 2:
# 			seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
# 		else:
# 			seg_result = np.zeros((patch_size[0], patch_size[1], patch_size[2]), dtype=np.float32)
#
# 	if dim == 2:
# 		data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
# 	else:
# 		data_result = np.zeros((patch_size[0], patch_size[1], patch_size[2]), dtype=np.float32)
#
# 	if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
# 		patch_center_dist_from_border = dim * [patch_center_dist_from_border]
#
#
# 	coords = create_zero_centered_coordinate_mesh(patch_size)
# 	modified_coords = False
#
# 	if do_elastic_deform and np.random.uniform() < p_el_per_sample:
# 		a = np.random.uniform(alpha[0], alpha[1])
# 		s = np.random.uniform(sigma[0], sigma[1])
# 		coords = elastic_deform_coordinates(coords, a, s)
# 		modified_coords = True
#
# 	if do_rotation and np.random.uniform() < p_rot_per_sample:
#
# 		if np.random.uniform() <= p_rot_per_axis:
# 			a_x = np.random.uniform(angle_x[0], angle_x[1])
# 		else:
# 			a_x = 0
#
# 		if dim == 3:
# 			if np.random.uniform() <= p_rot_per_axis:
# 				a_y = np.random.uniform(angle_y[0], angle_y[1])
# 			else:
# 				a_y = 0
#
# 			if np.random.uniform() <= p_rot_per_axis:
# 				a_z = np.random.uniform(angle_z[0], angle_z[1])
# 			else:
# 				a_z = 0
#
# 			coords = rotate_coords_3d(coords, a_x, a_y, a_z)
# 		else:
# 			coords = rotate_coords_2d(coords, a_x)
# 		modified_coords = True
#
# 	if do_scale and np.random.uniform() < p_scale_per_sample:
# 		if independent_scale_for_each_axis and np.random.uniform() < p_independent_scale_per_axis:
# 			sc = []
# 			for _ in range(dim):
# 				if np.random.random() < 0.5 and scale[0] < 1:
# 					sc.append(np.random.uniform(scale[0], 1))
# 				else:
# 					sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
# 		else:
# 			if np.random.random() < 0.5 and scale[0] < 1:
# 				sc = np.random.uniform(scale[0], 1)
# 			else:
# 				sc = np.random.uniform(max(scale[0], 1), scale[1])
#
# 		coords = scale_coords(coords, sc)
# 		modified_coords = True
#
# 	# now find a nice center location
# 	if modified_coords:
# 		for d in range(dim):
# 			if random_crop:
# 				ctr = np.random.uniform(patch_center_dist_from_border[d],
# 										data.shape[d] - patch_center_dist_from_border[d])
# 			else:
# 				ctr = data.shape[d] / 2. - 0.5
# 			coords[d] += ctr
# 		data_result = interpolate_img(data, coords, order_data, border_mode_data, cval=border_cval_data)
# 		if seg is not None:
# 			seg_result = interpolate_img(seg, coords, order_seg, border_mode_seg, cval=border_cval_seg, is_seg=True)
# 	else:
# 		if seg is None:
# 			s = None
# 		else:
# 			s = seg
# 		if random_crop:
# 			margin = [patch_center_dist_from_border[d] - patch_size[d] // 2 for d in range(dim)]
# 			d, s = random_crop_aug(data, s, patch_size, margin)
# 		else:
# 			d, s = center_crop_aug(data, patch_size, s)
# 		data_result = d
# 		if seg is not None:
# 			seg_result = s
# 	return data_result, seg_result

#对给定区域的三个方向（z轴、h轴、w轴）进行随机的平移（或称为"jittering"）
def augment_split_jittering(cursplit, curShapeOrg):
	# orgshape [z, h, w]
	#定区域的起始和结束位置，分别在z、h、w轴上。
	zstart, zend = cursplit[0][0], cursplit[0][1]
	hstart, hend = cursplit[1][0], cursplit[1][1]
	wstart, wend = cursplit[2][0], cursplit[2][1]
	#这些变量用于保存在z、h、w轴上的随机平移值。
	curzjitter, curhjitter, curwjitter = 0, 0, 0
	#这是平移的范围。当z轴上的范围小于等于3时，
	# 将范围设置为(zend - zstart) * 32，否则设置为(zend - zstart) * 2。
	if zend - zstart <= 3:
		jitter_range = (zend - zstart) * 32
	else:
		jitter_range = (zend - zstart) * 2
	# print("jittering range ", jitter_range)
	jitter_range_half = jitter_range//2

	t = 0
	#通过这个循环，尝试最多10次生成合适的平移值。
	while t < 10:
		if zstart == 0:
			curzjitter = int(np.random.rand() * jitter_range)
		elif zend == curShapeOrg[0]:
			curzjitter = -int(np.random.rand() * jitter_range)
		else:
			curzjitter = int(np.random.rand() * jitter_range) - jitter_range_half
		t += 1
		if (curzjitter + zstart >= 0) and (curzjitter + zend < curShapeOrg[0]):
			break

	t = 0
	while t < 10:
		if hstart == 0:
			curhjitter = int(np.random.rand() * jitter_range)
		elif hend == curShapeOrg[1]:
			curhjitter = -int(np.random.rand() * jitter_range)
		else:
			curhjitter = int(np.random.rand() * jitter_range) - jitter_range_half
		t += 1
		if (curhjitter + hstart >= 0) and (curhjitter + hend < curShapeOrg[1]):
			break

	t = 0
	while t < 10:
		if wstart == 0:
			curwjitter = int(np.random.rand() * jitter_range)
		elif wend == curShapeOrg[2]:
			curwjitter = -int(np.random.rand() * jitter_range)
		else:
			curwjitter = int(np.random.rand() * jitter_range) - jitter_range_half
		t += 1
		if (curwjitter + wstart >= 0) and (curwjitter + wend < curShapeOrg[2]):
			break

	if (curzjitter + zstart >= 0) and (curzjitter + zend < curShapeOrg[0]):
		cursplit[0][0] = curzjitter + zstart
		cursplit[0][1] = curzjitter + zend

	if (curhjitter + hstart >= 0) and (curhjitter + hend < curShapeOrg[1]):
		cursplit[1][0] = curhjitter + hstart
		cursplit[1][1] = curhjitter + hend

	if (curwjitter + wstart >= 0) and (curwjitter + wend < curShapeOrg[2]):
		cursplit[2][0] = curwjitter + wstart
		cursplit[2][1] = curwjitter + wend
	# print ("after ", cursplit)
	return cursplit


def augment(sample, label, coord=None, ifflip=True, ifswap=False, ifsmooth=False, ifjitter=False):
	"""
	:param sample, the cropped sample input
	:param label, the corresponding sample ground-truth
	:param coord, the corresponding sample coordinates
	:param ifflip, flag for random flipping
	:param ifswap, flag for random swapping
	:param ifsmooth, flag for Gaussian smoothing on the CT image
	:param ifjitter, flag for intensity jittering on the CT image
	:return: augmented training samples
	"""
	if ifswap:
		#设置为True，表示进行轴的置换。对于三维图像数据，可以随机调换坐标轴的顺序，以引入不同的视角。
		if sample.shape[0] == sample.shape[1] and sample.shape[0] == sample.shape[2]:
			#随机生成一个包含0、1、2的排列，表示对应三个轴的新顺序。
			axisorder = np.random.permutation(3)
			#根据生成的新顺序对sample进行轴置换操作。np.transpose函数用于交换数组的轴
			sample = np.transpose(sample, axisorder)
			label = np.transpose(label, axisorder)
			if coord is not None:
				#如果提供了坐标信息，同样对坐标信息进行相同的轴置换操作。
			# 坐标的轴顺序可能需要进行相应调整以匹配新的图像。
				coord = np.transpose(coord,np.concatenate([[0],axisorder+1]))
	#如果设置为True，表示进行随机翻转。通过在随机轴上翻转图像，引入镜像变换，以增加样本的多样性
	if ifflip:
		flipid = np.random.randint(2)*2-1
		sample = np.ascontiguousarray(sample[:,:,::flipid])
		label = np.ascontiguousarray(label[:,:,::flipid])
		if coord is not None:
			coord = np.ascontiguousarray(coord[:,:,:,::flipid])

	prob_aug = random.random()
	#ifjitter：如果设置为True，表示进行强度抖动。对CT图像进行强度抖动操作，以模拟不同强度的图像对比度。
	if ifjitter and prob_aug > 0.5:
		ADD_INT = (np.random.rand(sample.shape[0], sample.shape[1], sample.shape[2])*2 - 1)*10
		ADD_INT = ADD_INT.astype('float')
		cury_roi = label*ADD_INT/255.0
		sample += cury_roi
		sample[sample < 0] = 0
		sample[sample > 1] = 1

	prob_aug = random.random()
	#如果设置为True，表示进行高斯平滑。对CT图像进行高斯模糊操作，以模拟不同程度的图像模糊。
	if ifsmooth and prob_aug > 0.5:
		sigma = np.random.rand()
		if sigma > 0.5:
			sample = gaussian_filter(sample, sigma=1.0)

	return sample, label, coord
