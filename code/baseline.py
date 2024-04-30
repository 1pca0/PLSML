import torch
import torch.nn as nn
import numpy as np
from vnet import VNet
# from networks.magicnet import VNet_Magic
from networks.model_mj_in3sad import UNet3D
# Baseline

config = {'pad_value': 0,
          'augtype': {'flip': False, 'swap': False, 'smooth': False, 'jitter': False, 'split_jitter': False},
          'startepoch': 0, 'dataset_path': r'D:\daipeng\aircerebral_artery\process_data', 'dataset_split': r"D:\daipeng\semi_aricerebral\MT\semi_cerebral_0.2_0.8_1_2"}

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
	_, model = get_model()