import torch
import torch.nn as nn
import numpy as np
from vnet import VNet
# from networks.magicnet import VNet_Magic
from networks.model_mj_in3sad import UNet3D
# Baseline
import pickle
config = {'pad_value': 0,
          'augtype': {'flip': True, 'swap': False, 'smooth': False, 'jitter': True, 'split_jitter': True},
          'startepoch': 0, 'dataset_path': r'D:\daipeng\supervised-full\preprocessed_datasets', 'dataset_split': r"D:\daipeng\MCF+reliable_label\code\train_MCF_airway\semi_BAS_0.1_0.9_1_2"}

def get_model(args=None,ema=False):
	#net = VNet_Magic(1,2,32,96,16,normalization='instancenorm')
	net = UNet3D(1,2,coord=False)
	print(net)
	#这里出现网络参数
	if ema:
		for param in net.parameters():
			param.detach_()
	print('# of network parameters:', sum(param.numel() for param in net.parameters()))
	return config, net


if __name__ == '__main__':
	# _, model = get_model()
	with open(r"D:\daipeng\MCF+reliable_label\code\train_MCF_airway\semi_BAS_0.1_0.9_1_2","rb") as datas:
		data = pickle.load(datas)
		trainslabel = data['train']['label']
		trainsunlabel = data['train']['unlabel']
		vals = data['val']
		tests = data['test']
		print(len(trainslabel))
		print(len(trainsunlabel))
		print(len(tests))
		print(trainsunlabel)
		print(tests)

