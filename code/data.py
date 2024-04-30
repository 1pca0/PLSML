import random

import  numpy as  np
import torch
from torch.utils.data import  Dataset
import os
import time
from math import sqrt
from torchvision import transforms
import SimpleITK as sitk
from glob import glob
from scipy.ndimage import gaussian_filter
from scipy.special import erfinv
from utils import save_itk ,load_itk_image ,lumTrans,load_pickle

from transform import crop ,hflip,normalize,resize,blur
import  itertools
from torch.utils.data.sampler import Sampler
import math

def relabel_dataset(dataset):
    unlabeled_idxs = []
    labeled_idxs= []
    # print(dataset.unlabeled_name)
    # print("###########################")
    #print(f"长度为{len(dataset.cubelist)}")
    for idx in range (len(dataset.cubelist)):

        xx = dataset.cubelist[idx][0].split('\\')[-1]
        # print(dataset.cubelist[idx][0])
        if xx in dataset.unlabeled_name :
            # print(f"进来了")
            unlabeled_idxs.append(idx)
        else:
            labeled_idxs.append(idx)

    # labeled_idxs = sorted(set(range(len(dataset))) -set(unlabeled_idxs))

    #unlabeled_idxs = unlabeled_idxs *math.ceil(len(labeled_idxs) / len(unlabeled_idxs))
    # labeled_idxs = labeled_idxs * math.ceil(len(unlabeled_idxs) / len(labeled_idxs))
    return labeled_idxs,unlabeled_idxs

class TwoStreamBatchSampler(Sampler):
    def __init__(self,primary_indices,secondary_indices,batch_size,secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size


    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_once(self.secondary_indices)

        return (
            primary_batch +secondary_batch
            for (primary_batch ,secondary_batch)
            in zip(grouper(primary_iter,self.primary_batch_size),
                   grouper(secondary_iter,self.secondary_batch_size))
        )

    def __len(self):
        return len(self.primary_indices)//self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)



def grouper(iterable,n):
    args = [iter(iterable)] *n
    return zip(*args)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())

#傅里叶变换
# class CollateClass(object):
#     def __init__(self,FF):
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
		# print(f"长度为{len(sample_list)}")
		for i in range(len(sample_list)):
			x.append(sample_list[i][0])
			y.append(sample_list[i][1])
			coord.append(sample_list[i][2])
		#  label =
		# x1, x2 = x[:bs]
		# x21, x12 = colorful_spectrum_mix(np.squeeze(x1), np.squeeze(x2), self.alpha, ratio=1.0)
		# x21 = x21[np.newaxis, ...]
		# x12 = x12[np.newaxis, ...]
		if self.FFT:
			# print('FFT')
			x1 = np.stack((x[:bs]), axis=0)
			y1 = np.stack((y[:]), axis=0)
			coord1 = np.stack((coord[:bs]), axis=0)
			img1, img2 = x[bs:]
			x3 = np.stack((img1, img2), axis=0)

			start_time = time.time()
			img21, img12 = colorful_spectrum_mix(np.squeeze(img1), np.squeeze(img2), self.alpha, ratio=1.0, matrix=self.matrix)
			end_time =time.time()
			print(f"傅里叶变换时间{end_time-start_time}")

			del img1, img2
			img21 = img21[np.newaxis, ...]
			img12 = img12[np.newaxis, ...]


			# img211 = img211[np.newaxis, ...]
			# img122 = img122[np.newaxis, ...]

			x2 = np.stack((img21, img12), axis=0)
			del img21, img12

			new_x = np.concatenate((x1, x2, x3), axis=0)

			del x1, x2, x3
			coord2 = np.stack((coord[bs:]), axis=0)
			new_coord = np.concatenate((coord1, coord2, coord2), axis=0)
			del coord1, coord2
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

class AirwayData(Dataset):
    def __init__(self,config,args,split_comber =None ,debug =False):
        self.augtype = config['augtype']
        self.split_comber = split_comber
        self.debug_flag = debug
        self.datapath = config['dataset_path']

        self.dataset  = load_pickle(config['dataset_split'])

class train_datset(AirwayData):
    def __init__(self,config,args,split_comber = None,debug=False):
        super(train_datset,self).__init__(config,args,split_comber = split_comber,debug = debug)

        print("-----------------------------load train data into memory----------------------------")

        cubelist = []
        self.caseNumber = 0
        allimgdata_memory ={}
        alllabeldata_memory = {}

        file_names = self.dataset['train']
        data_file_names = file_names['train']['label']
        unlabel_data_file_names = file_names['train']['unlabel']
        file_num = len(data_file_names) + len(unlabel_data_file_names)
        if self.debug_flag:
            data_file_names = file_names[:1]
            file_num = len(data_file_names)
        self.caseNumber += file_num

        print("total %s case number %d "% ('train',self.caseNumber))

        for raw_path in data_file_names:

            raw_path = os.path.join(self.datapath,raw_path.split('/')[-1]+'.gz')
            assert (os.path.exists(raw_path) is True)
            label_path = raw_path.replace('clean_hu','label')
            assert (os.path.exists(label_path) is True)

            imgs,origin,spacing = load_itk_image(raw_path)
            splits,nzhw,orgshape  = self.split_comber.split_id(imgs)
            data_name =raw_path.split('/')[-1].split('_clean_hu')[0]

            print("Name:%s,# of splits :%d"% (data_name,len(splits)))
            labels,_,_ = load_itk_image(label_path)

            allimgdata_memory[data_name] = [imgs,origin,spacing]
            alllabeldata_memory[data_name] = labels
            cube_train = []

            for j in range(len(splits)):

                cursplit = splits[j]
                labelcube = labels[cursplit[0][0]:cursplit[0][1],cursplit[1][0]:cursplit[1][1],cursplit[2][0]:cursplit[2][1]]

                curnumlabel = np.sum(labelcube)

                if curnumlabel >0 :
                    curlist = [data_name,cursplit,j,nzhw,orgshape,"Y"]
                    cube_train.append(curlist)

            random.shuffle(cube_train)
            cubelist +=cube_train
        for raw_path in unlabel_data_file_names:

            raw_path = os.path.join(self.datapath,raw_path.split('/')[-1]+'.gz')
            assert (os.path.exists(raw_path) is True)
            label_path = raw_path.replace('clean_hu','label')
            assert (os.path.exists(label_path) is True)

            imgs,origin,spacing = load_itk_image(raw_path)
            splits,nzhw,orgshape  = self.split_comber.split_id(imgs)
            data_name =raw_path.split('/')[-1].split('_clean_hu')[0]

            print("Name:%s,# of splits :%d"% (data_name,len(splits)))
            labels,_,_ = load_itk_image(label_path)

            allimgdata_memory[data_name] = [imgs,origin,spacing]
            alllabeldata_memory[data_name] = labels
            cube_train = []

            for j in range(len(splits)):

                cursplit = splits[j]
                labelcube = labels[cursplit[0][0]:cursplit[0][1],cursplit[1][0]:cursplit[1][1],cursplit[2][0]:cursplit[2][1]]

                curnumlabel = np.sum(labelcube)

                if curnumlabel >0 :
                    curlist = [data_name,cursplit,j,nzhw,orgshape,"Y"]
                    cube_train.append(curlist)

            random.shuffle(cube_train)
            cubelist +=cube_train

        self.allimgdata_memory = allimgdata_memory
        self.alllabeldata_memory  =alllabeldata_memory

        random.shuffle(cubelist)
        self.cubelist = cubelist

        print("------------------------------Initialization Done----------------------------")

        print("Phase: %s total cubelist number: %d "% ('train',len(self.cubelist)))

        print()

    def __len__(self):
        return len(self.cubelist)

    def __getitem__(self, idx):
        t  = time.time()
        np.random.seed(int(str(t%1)[2:7]))
        curlist = self.cubelist[idx]

        curNameID = curlist[0]
        cursplit = curlist[1]
        curShapeOrg = curlist[4]
        curtransFlag = curlist[5]

    ######################################################

        imginfo = self.allimgdata_memory[curNameID]
        imgs,origin,spacing = imginfo[0],imginfo[1],imginfo[2]
        curcube = imgs[cursplit[0][0]:cursplit[0][1] ,cursplit[1][0]:cursplit[1][1],cursplit[2][0]:cursplit[2][1]]
        curcube = (curcube.astype(np.float32))/255.0

    #########################################################

        start = [float(cursplit[0][0]),float(cursplit[1][0]),float(cursplit[2][0])]
        normstart = ((np.array(start).astype('float')/ np.array(curShapeOrg).astype('float')) - 0.5) *2.0
        crop_size = [curcube.shape[0],curcube.shape[1],curcube.shape[2]]

        stride = 1.0
        normsize = (np.array(crop_size).astype('float') / np.array(curShapeOrg).astype('float')) *2.0

        xx,yy,zz = np.meshgrid(np.linspace(normstart[0],normstart[0]+normsize[0],int(crop_size[0])),
                               np.linspace(normstart[1],normstart[1]+normsize[1],int(crop_size[1])),
                               np.linspace(normstart[2],normstart[2]+normsize[2],int(crop_size[2])),
                               indexing = 'ij'
                               )

        coord  = np.concatenate([xx[np.newaxis,...],yy[np.newaxis,...],zz[np.newaxis,...]],0).astype('float')

        assert (coord.shape[0]== 3)


        label =self.alllabeldata_memory[curNameID]
        label = label[cursplit[0][0]:cursplit[0][1],cursplit[1][0]:cursplit[1][1],cursplit[2][0]:cursplit[2][1]]
        label =(label>0)
        label = label.astype('float')


    ###################################################################

        if curtransFlag == "Y":
            curcube,label ,coord = augment(curcube,label,coord,ifflip = self.augtype['flip'],ifswap = self.augtype['swap'],ifsmooth =self.augtype['smooth'],ifjitter=self.augtype['jitter'])

        curcube =curcube[np.newaxis,...]
        label = label[np.newaxis,...]
        x = torch.from_numpy(curcube).float()
        y =torch.from_numpy(curcube).float()

        coord = torch.from_numpy(coord).float()

        return x ,y,coord



class semi_train_dataset(AirwayData):

    def __init__(self,config,args,split_comber =None,debug =False):

        super (semi_train_dataset,self).__init__(config,args,split_comber =split_comber,debug = debug)

        self.augment_spatial = args.augment_spatial
        self.RD  =args.elastic_deform
        self.rotation = args.rotation
        self.scale = args.scale
        self.rand_crop =args.rand_crop
        self.colorjitter = args.colorjitter
        self.cutout = args.cut
        self.n =args.n_holes
        self.crop = args.cutout
        self.FFT = args.fft
        self.alpha =args.alpha
        self.sigma_min = args.sigma_min
        self.sigma_max =args.sigma_max
        self.p_min = args.p_min
        self.p_max =args.p_max
        self.unlabeled_name =[]

        unlabel_data_file_names = self.dataset['train']['unlabel']
        #print(f"unlabel_data_file_names{unlabel_data_file_names}")
        for i in unlabel_data_file_names:

            i = os.path.join(self.datapath,i.split('/')[-1])
            assert (os.path.join(self.datapath,i.split('/')[-1]))
            label_path = i.replace('clean_hu','label')
            #print(f"label_path{label_path}")
            assert (os.path.exists(label_path) is True)
            # self.unlabeled_name.append(i.split('/')[-1].split('_clean_hu')[0])
            self.unlabeled_name.append(i.split("\\")[-1].split('_clean_hu')[0])



        print("----------------------------------------load all semi train data into memory-----------------------------")

        cubelist = []
        self.caseNumber = 0
        allimgdata_memory = {}
        alllabeldata_memory = {}
        file_names = self.dataset['train']

        data_file_names = file_names['label']
        # print(f"{data_file_names}")
        # print("############################################")
        unlabel_data_file_names = file_names['unlabel']
        file_num = len(data_file_names) +len(unlabel_data_file_names)

        if self.debug_flag:
            data_file_names = file_names['label'][:4]
            file_num = len(data_file_names)

        self.caseNumber += file_num


        print("total %s casenumber :%d"%('semi_train',self.caseNumber))

        for raw_path in data_file_names:
            name = raw_path.split('/')[-1].split('_clean_hu')[0]

            raw_path = os.path.join(self.datapath,raw_path.split('/')[-1])
            assert(os.path.exists(raw_path) is True)
            label_path = raw_path.replace('clean_hu','label')
            assert(os.path.exists(label_path) is True)

            imgs ,origin,spacing = load_itk_image(raw_path)

            splits,nzhw,orgshape =self.split_comber.split_id(imgs)
            data_name = raw_path.split('/')[-1].split('_clean_hu')[0]

            print("Name: %s,# of splits :%d"%(data_name,len(splits)))
            labels,_,_ =load_itk_image(label_path)

            allimgdata_memory[data_name] = [imgs,origin,spacing]
            alllabeldata_memory[data_name] = labels

            cube_train = []
            for j in range (len(splits)):

                cursplit = splits[j]
                labelcube = labels[cursplit[0][0]:cursplit[0][1],cursplit[1][0]:cursplit[1][1],cursplit[2][0]:cursplit[2][1]]
                curnumlabel = np.sum(labelcube)

                if curnumlabel > 0:
                    curlist = [data_name,cursplit,j ,nzhw,orgshape ,"Y"]
                    cube_train.append(curlist)

            random.shuffle(cube_train)
            cubelist +=cube_train
        for raw_path in unlabel_data_file_names:
            name = raw_path.split('/')[-1].split('_clean_hu')[0]

            raw_path = os.path.join(self.datapath,raw_path.split('/')[-1] )
            assert(os.path.exists(raw_path) is True)
            label_path = raw_path.replace('clean_hu','label')
            assert(os.path.exists(label_path) is True)

            imgs ,origin,spacing = load_itk_image(raw_path)

            splits,nzhw,orgshape =self.split_comber.split_id(imgs)
            data_name = raw_path.split('/')[-1].split('_clean_hu')[0]

            print("Name: %s,# of splits :%d"%(data_name,len(splits)))
            labels,_,_ =load_itk_image(label_path)

            allimgdata_memory[data_name] = [imgs,origin,spacing]
            alllabeldata_memory[data_name] = labels

            #name = os.path.join(r"D:\\daipeng\\airway_segemntation_temp\\preprocessed_data",name)
            if name in self.unlabeled_name:
                for j in range(len(splits)):
                    # print(f"进来了")
                    cursplit = splits[j]
                    curlist = [data_name,cursplit,j,nzhw,orgshape,'Y']

                    cubelist.append(curlist)


        self.allimgdata_memory = allimgdata_memory
        self.alllabeldata_memory = alllabeldata_memory

        random.shuffle(cubelist)
        self.cubelist = cubelist

        print("---------------------------------------Initialization Done ------------------------------------")

        print('Phase: %s total cubelist number:%d '%('semi_train',len(self.cubelist)))

        print()

    def __len__(self):
        return len(self.cubelist)

    def __getitem__(self, idx):


        t =time.time()
        np.random.seed(int(str(t%1)[2:7]))
        curlist = self.cubelist[idx]


        curNameID =curlist[0]
        cursplit = curlist[1]

        curShapeOrg = curlist[4]

        curtransFlag = curlist[5]

        if curtransFlag == 'Y' and self.augtype['split_jitter'] is True:

            cursplit = augment_split_jittering(cursplit,curShapeOrg)

        ####################################

        imginfo = self.allimgdata_memory[curNameID]

        imgs,origin ,spacing = imginfo[0],imginfo[1],imginfo[2]

        curcube = imgs[cursplit[0][0]:cursplit[0][1],cursplit[1][0]:cursplit[1][1],cursplit[2][0]:cursplit[2][1]]

        curcube = (curcube.astype(np.float32))/255.0


        ########################################

        start = [float(cursplit[0][0]),float(cursplit[1][0]),float(cursplit[2][0])]
        normstart = ((np.array(start).astype('float')/np.array(curShapeOrg).astype('float')) -0.5)*2.0
        crop_size = [curcube.shape[0],curcube.shape[1],curcube.shape[2]]
        stride = 1.0
        normsize= (np.array(crop_size).astype('float')/np.array(curShapeOrg).astype('float')) *2.0

        xx ,yy,zz = np.meshgrid(np.linspace(normstart[0],normstart[0]+normsize[0],int(crop_size[0])),
                                np.linspace(normstart[1],normstart[1]+normsize[1],int(crop_size[1])),
                                np.linspace(normstart[2],normstart[2]+normsize[2],int(crop_size[2])),
                                indexing= 'ij'
                                )

        coord  =np.concatenate([xx[np.newaxis,...],yy[np.newaxis,...],zz[np.newaxis,...]],0).astype('float')

        assert (coord.shape[0] ==3)


        label =self.alllabeldata_memory[curNameID]
        label =label[cursplit[0][0]:cursplit[0][1],cursplit[1][0]:cursplit[1][1],cursplit[2][0]:cursplit[2][1]]
        label = (label >0)
        label = label.astype('float')


        if  curtransFlag =='Y':
            curcube,label ,coord =augment(curcube,label,coord,ifflip = self.augtype['flip'],ifswap = self.augtype['swap'],
                                              ifsmooth = self.augtype['smooth'],ifjitter = self.augtype['jitter']

                                              )

            x = curcube[np.newaxis,...]
            y =label[np.newaxis,...]

        return x,y,coord


class val_dataset(AirwayData):

    def __init__(self,config,args,split_comber =None,debug =False):
        super(val_dataset,self).__init__(config,args,split_comber = split_comber,debug =debug)

        print("------------------------------------load all val data into memory------------------------------")


        cubelist = []
        self.caseNumber  =0
        allimgdata_memory = {}
        alllabeldata_memory = {}

        file_names = self.dataset['val']
        data_file_names = file_names
        file_num = len(data_file_names)

        if self.debug_flag:
            data_file_names = file_names[:]
            file_num = len(data_file_names)

        self.caseNumber +=file_num
        print("total %s case number:%d" %('val',self.caseNumber))

        for raw_path in data_file_names :
            raw_path = os.path.join(self.datapath,raw_path.split('/')[-1])

            assert (os.path.exists(raw_path) is True)

            label_path = raw_path.replace('clean_hu','label')

            assert (os.path.exists(label_path) is True)
            imgs,origin ,spacing = load_itk_image(raw_path)
            splits,nzhw,orgshape = self.split_comber.split_id(imgs)
            data_name = raw_path.split('/')[-1].split('_clean_hu')[0]
            print("Name:%s ,#of splits :%d" %(data_name,len(splits)))

            labels,_,_,= load_itk_image(label_path)

            allimgdata_memory[data_name] = [imgs,origin,spacing]
            alllabeldata_memory[data_name] = labels


            for j in range(len(splits)):
                cursplit = splits[j]
                curlist = [data_name,cursplit,j,nzhw,orgshape,'N']
                cubelist.append(curlist)

        self.allimgdata_memory = allimgdata_memory
        self.alllabeldata_memory = alllabeldata_memory
        self.cubelist = cubelist

        print("--------------------------------Initialization Done------------------------")
        print("Phase: %s totaal cubelist number :%d" %('val',len(self.cubelist)))
        print()

    def __len__(self):
        return len(self.cubelist)

    def __getitem__(self, idx):

        t =time.time()
        np.random.seed(int(str(t %1)[2:7]))
        curlist = self.cubelist[idx]


        curNameID = curlist[0]
        cursplit = curlist[1]

        curSplitID = curlist[2]
        curnzhw  = curlist[3]
        curShapeOrg = curlist[4]

        ###############################
        imginfo = self.allimgdata_memory[curNameID]
        imgs,origin ,spacing = imginfo[0],imginfo[1],imginfo[2]

        curcube = imgs[cursplit[0][0]:cursplit[0][1],cursplit[1][0]:cursplit[1][1],cursplit[2][0]:cursplit[2][1]]
        curcube = (curcube.astype(np.float32))/255.0

        #####################################################
        start =[float(cursplit[0][0]),float(cursplit[1][0]),float(cursplit[2][0])]

        normstart = ((np.array(start).astype('float')/np.array(curShapeOrg).astype('float'))-0.5)*2.0
        crop_size = [curcube.shape[0],curcube.shape[1],curcube.shape[2]]
        stride = 1.0
        normasize = (np.array (crop_size).astype('float')/np.array(curShapeOrg).astype('float')) *2.0

        xx ,yy ,zz = np.meshgrid(np.linspace(normstart[0],normstart[0]+normasize[0],int(crop_size[0])),
                                 np.linspace(normstart[1], normstart[1] + normasize[1], int(crop_size[1])),
                                 np.linspace(normstart[2], normstart[2] + normasize[2], int(crop_size[2])),
                                 )
        coord = np.concatenate([xx[np.newaxis,...],yy[np.newaxis,...],zz[np.newaxis,...]],0).astype('float')
        assert (coord.shape[0] ==3)

        label =self.alllabeldata_memory[curNameID]
        label =label [cursplit[0][0]:cursplit[0][1],cursplit[1][0]:cursplit[1][1],cursplit[2][0]:cursplit[2][1]]
        label =(label >0)

        label = label.astype('float')

        curnzhw = np.array(curnzhw)
        curShapeOrg = np.array(curShapeOrg)

        curcube = curcube[np.newaxis,...]
        label = label [np.newaxis,...]

        return torch.from_numpy(curcube).float(),torch.from_numpy(label).float(),\
                torch.from_numpy(coord).float(),torch.from_numpy(origin),\
                torch.from_numpy(spacing),curNameID,curSplitID,\
                torch.from_numpy(curnzhw),torch.from_numpy(curShapeOrg)

class test_dataset(AirwayData):

    def __init__(self,config,args,split_comber =None,debug =False):
        super(test_dataset,self).__init__(config,args,split_comber = split_comber,debug =debug)

        print("------------------------------------load all val data into memory------------------------------")


        cubelist = []
        self.caseNumber  =0
        allimgdata_memory = {}
        alllabeldata_memory = {}

        file_names = self.dataset['test']
        data_file_names = file_names
        file_num = len(data_file_names)

        if self.debug_flag:
            data_file_names = file_names[:]
            file_num = len(data_file_names)

        self.caseNumber +=file_num
        print("total %s case number:%d" %('test',self.caseNumber))

        for raw_path in data_file_names :
            raw_path = os.path.join(self.datapath,raw_path.split('/')[-1])

            assert (os.path.exists(raw_path) is True)

            label_path = raw_path.replace('clean_hu','label')

            assert (os.path.exists(label_path) is True)
            imgs,origin ,spacing = load_itk_image(raw_path)
            splits,nzhw,orgshape = self.split_comber.split_id(imgs)
            data_name = raw_path.split('/')[-1].split('_clean_hu')[0]
            print("Name:%s ,#of splits :%d" %(data_name,len(splits)))

            labels,_,_,= load_itk_image(label_path)

            allimgdata_memory[data_name] = [imgs,origin,spacing]
            alllabeldata_memory[data_name] = labels


            for j in range(len(splits)):
                cursplit = splits[j]
                curlist = [data_name,cursplit,j,nzhw,orgshape,'N']
                cubelist.append(curlist)

        self.allimgdata_memory = allimgdata_memory
        self.alllabeldata_memory = alllabeldata_memory
        self.cubelist = cubelist

        print("--------------------------------Initialization Done------------------------")
        print("Phase: %s totaal cubelist number :%d" %('val',len(self.cubelist)))
        print()

    def __len__(self):
        return len(self.cubelist)

    def __getitem__(self, idx):

        t =time.time()
        np.random.seed(int(str(t %1)[2:7]))
        curlist = self.cubelist[idx]


        curNameID = curlist[0]
        cursplit = curlist[1]

        curSplitID = curlist[2]
        curnzhw  = curlist[3]
        curShapeOrg = curlist[4]

        ###############################
        imginfo = self.allimgdata_memory[curNameID]
        imgs,origin ,spacing = imginfo[0],imginfo[1],imginfo[2]

        curcube = imgs[cursplit[0][0]:cursplit[0][1],cursplit[1][0]:cursplit[1][1],cursplit[2][0]:cursplit[2][1]]
        curcube = (curcube.astype(np.float32))/255.0

        #####################################################
        start =[float(cursplit[0][0]),float(cursplit[1][0]),float(cursplit[2][0])]

        normstart = ((np.array(start).astype('float')/np.array(curShapeOrg).astype('float'))-0.5)*2.0
        crop_size = [curcube.shape[0],curcube.shape[1],curcube.shape[2]]
        stride = 1.0
        normasize = (np.array (crop_size).astype('float')/np.array(curShapeOrg).astype('float')) *2.0

        xx ,yy ,zz = np.meshgrid(np.linspace(normstart[0],normstart[0]+normasize[0],int(crop_size[0])),
                                 np.linspace(normstart[1], normstart[1] + normasize[1], int(crop_size[1])),
                                 np.linspace(normstart[2], normstart[2] + normasize[2], int(crop_size[2])),
                                 )
        coord = np.concatenate([xx[np.newaxis,...],yy[np.newaxis,...],zz[np.newaxis,...]],0).astype('float')
        assert (coord.shape[0] ==3)

        label =self.alllabeldata_memory[curNameID]
        label =label [cursplit[0][0]:cursplit[0][1],cursplit[1][0]:cursplit[1][1],cursplit[2][0]:cursplit[2][1]]
        label =(label >0)

        label = label.astype('float')

        curnzhw = np.array(curnzhw)
        curShapeOrg = np.array(curShapeOrg)

        curcube = curcube[np.newaxis,...]
        label = label [np.newaxis,...]

        return torch.from_numpy(curcube).float(),torch.from_numpy(label).float(),\
                torch.from_numpy(coord).float(),torch.from_numpy(origin),\
                torch.from_numpy(spacing),curNameID,curSplitID,\
                torch.from_numpy(curnzhw),torch.from_numpy(curShapeOrg)
def augment_split_jittering(cursplit,curShapeOrg):
    zstart ,zend =cursplit[0][0],cursplit[0][1]
    hstart, hend = cursplit[1][0],cursplit[1][1]
    wstart,wend = cursplit[2][0],cursplit[2][1]

    curzjitter ,curhjitter ,curwjitter = 0 ,0 ,0
    if zend -zstart <=3:
        jitter_range = (zend - zstart) *32
    else:
        jitter_range = (zend -zstart)*2

    jitter_range_half = jitter_range//2

    t = 0
    while t<10 :
        if zstart ==0 :
            curzjitter = int(np.random.rand() *jitter_range)
        elif zend ==curShapeOrg[0]:
            curzjitter = -int(np.random.rand() *jitter_range)
        else:
            curzjitter = int(np.random.rand() * jitter_range) - jitter_range_half
        t+=1
        if (curzjitter + zstart >=0) and(curzjitter +zend <curShapeOrg[0]):
            break

    t =0
    while t<10:
        if hstart ==0 :
            curhjitter = int(np.random.rand() *jitter_range)
        elif zend ==curShapeOrg[0]:
            curhjitter = -int(np.random.rand() *jitter_range)
        else:
            curhjitter = int(np.random.rand() * jitter_range) - jitter_range_half
        t+=1
        if (curhjitter + hstart >=0) and(curhjitter +hend <curShapeOrg[1]):
            break

    t = 0
    while t < 10:
        if wstart == 0:
            curwjitter = int(np.random.rand() * jitter_range)
        elif wend == curShapeOrg[0]:
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
		if sample.shape[0] == sample.shape[1] and sample.shape[0] == sample.shape[2]:
			axisorder = np.random.permutation(3)
			sample = np.transpose(sample, axisorder)
			label = np.transpose(label, axisorder)
			if coord is not None:
				coord = np.transpose(coord,np.concatenate([[0],axisorder+1]))

	if ifflip:
		flipid = np.random.randint(2)*2-1
		sample = np.ascontiguousarray(sample[:,:,::flipid])
		label = np.ascontiguousarray(label[:,:,::flipid])
		if coord is not None:
			coord = np.ascontiguousarray(coord[:,:,:,::flipid])

	prob_aug = random.random()
	if ifjitter and prob_aug > 0.5:
		ADD_INT = (np.random.rand(sample.shape[0], sample.shape[1], sample.shape[2])*2 - 1)*10
		ADD_INT = ADD_INT.astype('float')
		cury_roi = label*ADD_INT/255.0
		sample += cury_roi
		sample[sample < 0] = 0
		sample[sample > 1] = 1

	prob_aug = random.random()
	if ifsmooth and prob_aug > 0.5:
		sigma = np.random.rand()
		if sigma > 0.5:
			sample = gaussian_filter(sample, sigma=1.0)

	return sample, label, coord

# class test_dataset(AirwayData):
#     def __init__(self,config,args,split_comber= None,debug= False):
#         super(test_dataset,self).__init__(config,args,split_comber= split_comber,debug =debug)
#
#         print("---------------------------------Load all test data into memory----------------------------")
#
#         cubelist = []
#         self.caseNumber = 0
#         allimgdata_memory = {}
#         alllabeldata_memory = {}
#         file_names = self.dataset['test']
#         data_file_names = file_names
#         file_num = len(data_file_names)
#         if self.debug_flag:
#             data_file_names = file_names[:1]
#             file_num = len(data_file_names)
#
#         self.caseNumber +=file_num
#         print("total %s case number:%d"%('test',self.caseNumber))
#
#         for raw_path in data_file_names:
#             raw_path = os.path.join(self.datapath,raw_path.split('/')[-1])
#             print(raw_path)
#             assert (os.path.exists(raw_path) is True)
#
#             label_path = raw_path.replace('clean_hu','label')
#             assert (os.path.exists(label_path) is True)
#
#             imgs,origin,spacing = load_itk_image(raw_path)
#             splits,nzhw,orgshape = self.split_comber.split_id(imgs)
#             data_name = raw_path.split('/')[-1].split('_clean_hu')[0]
#
#             print ("Name;%s,#of splits:%d"% (data_name,len(splits)))
#             labels ,_,_ = load_itk_image(label_path)
#
#             allimgdata_memory[data_name] = [imgs,origin,spacing]
#             alllabeldata_memory[data_name] = labels
#
#             for j in range(len(splits)):
#
#                 cursplit = splits[j]
#                 curlist = [data_name,cursplit,j,nzhw,orgshape,"N"]
#                 cubelist.append(curlist)
#
#         self.allimgdata_memory = allimgdata_memory
#         self.alllabeldata_memory = alllabeldata_memory
#
#         self.cubelist = cubelist
#
#         print('------------------------------Initialization Done-----------------------------')
#
#         print('Phase:%stotal cubelist number :%d'% ('test',len(self.cubelist)))
#
#         print()
#
#     def __len__(self):
#         return len(self.cubelist)
#
#     def __getitem__(self, idx):
#         t =time.time()
#         np.random.seed(int(str(t %1)[2:7]))
#         curlist = self.cubelist[idx]
#
#         curNameID = curlist[0]
#         cursplit = curlist[1]
#         curSplitId = curlist[2]
#         curnzhw = curlist[3]
#         curShapeOrg = curlist[4]
#
#         ###############################################
#         imginfo =  self.allimgdata_memory[curNameID]
#
#         imgs,origin,spacing = imginfo[0],imginfo[1],imginfo[2]
#         curcube = imgs[cursplit[0][0]:cursplit[0][1], cursplit[1][0]:cursplit[1][1], cursplit[2][0]:cursplit[2][1]]
#         curcube = (curcube.astype(np.float32))/255.0

        #########################################


















































































































































































































































































































































































































