import numpy as np
import os
import sys
import time
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from model import *

if __name__ == '__main__':
	data_path = sys.argv[1]
	current_fold = int(sys.argv[2])
	organ_number = int(sys.argv[3])
	low_range = int(sys.argv[4])
	high_range = int(sys.argv[5])
	slice_threshold = float(sys.argv[6])
	slice_thickness = int(sys.argv[7])
	organ_ID = int(sys.argv[8])
	plane = sys.argv[9]
	GPU_ID = int(sys.argv[10])
	learning_rate1 = float(sys.argv[11])
	learning_rate_m1 = int(sys.argv[12])
	learning_rate2 = float(sys.argv[13])
	learning_rate_m2 = int(sys.argv[14])
	crop_margin = int(sys.argv[15])
	crop_prob = float(sys.argv[16])
	crop_sample_batch = int(sys.argv[17])
	snapshot_path = os.path.join(snapshot_path, 'SIJ_training_' + \
		sys.argv[11] + 'x' + str(learning_rate_m1) + ',' + str(crop_margin))
	step = int(sys.argv[18])
	max_iterations1 = int(sys.argv[19]) # 40000
	max_iterations2 = int(sys.argv[20])
	fraction = float(sys.argv[21]) # 0.25
	separate_iterations = max(int(round(max_iterations1 * fraction / step)), 1) * step
	timestamp = sys.argv[22]

	if not os.path.exists(snapshot_path):
		os.makedirs(snapshot_path)
	
	FCN_weights = os.path.join(pretrained_model_path, 'fcn8s_from_caffe.pth')
	if not os.path.isfile(FCN_weights):
		raise RuntimeError('Please Download <http://drive.google.com/uc?id=0B9P1L--7Wd2vT0FtdThWREhjNkU> from the Internet ...')

	from Data import DataLayer
	training_set = DataLayer(data_path=data_path, current_fold=current_fold, organ_number=organ_number, \
		low_range=low_range, high_range=high_range, slice_threshold=slice_threshold, slice_thickness=slice_thickness, \
		organ_ID=organ_ID, plane=plane)

	batch_size = 1
	os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_ID)
	trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=16)
	max_iterations = {'S': separate_iterations, \
					  'I': max_iterations1 - separate_iterations, \
					  'J': max_iterations1 }
	print(plane, len(trainloader))
	print(max_iterations)
	

	RSTN_ = {}
	RSTN_snapshot = {}
	for mode in ['S', 'I', 'J']:
		RSTN_[mode] = RSTN(mode=mode, crop_margin=crop_margin, \
							crop_prob=crop_prob, crop_sample_batch=crop_sample_batch)
		RSTN_dict = RSTN_[mode].state_dict()
		print('RSTN_' + mode + '_dict', RSTN_dict.keys())

		if mode == 'S':
			pretrained_model = FCN8s(n_class=21)
			pretrained_model.load_state_dict(torch.load(FCN_weights))
			pretrained_dict = pretrained_model.state_dict()
			# 1. filter out unnecessary keys
			pretrained_dict_coarse = {'coarse_model.' + k : v
					for k, v in pretrained_dict.items() 
					if 'coarse_model.' + k in RSTN_dict and 'score' not in k}
			pretrained_dict_fine = {'fine_model.' + k : v
					for k, v in pretrained_dict.items() 
					if 'fine_model.' + k in RSTN_dict and 'score' not in k}
			# 2. overwrite entries in the existing state dict
			RSTN_dict.update(pretrained_dict_coarse) 
			RSTN_dict.update(pretrained_dict_fine)
			# 3. load the new state dict
			RSTN_[mode].load_state_dict(RSTN_dict)
		
		elif mode == 'I':
			RSTN_[mode].load_state_dict(torch.load(RSTN_snapshot['S']))
		elif mode == 'J':
			RSTN_[mode].load_state_dict(torch.load(RSTN_snapshot['I']))
		else:
			raise ValueError("wrong value of mode, should be in ['S', 'I', 'J']")

		print(plane + mode, 'load pre-trained model successfully!')
		model_parameters = filter(lambda p: p.requires_grad, RSTN_[mode].parameters())
		params = sum([np.prod(p.size()) for p in model_parameters])
		print('model parameters:', params)

		optimizer = torch.optim.SGD(
			[
				{'params': get_parameters(RSTN_[mode], coarse=True, bias=False)},
				{'params': get_parameters(RSTN_[mode], coarse=True, bias=True),
				'lr': learning_rate1 * 2, 'weight_decay': 0},
				{'params': get_parameters(RSTN_[mode], coarse=False, bias=False),
				'lr': learning_rate1 * 10},
				{'params': get_parameters(RSTN_[mode], coarse=False, bias=True),
				'lr': learning_rate1 * 20, 'weight_decay': 0}	
			],
			lr=learning_rate1,
			momentum=0.99,
			weight_decay=0.0005)

		criterion = DSC_loss()
		COARSE_WEIGHT = 1 / 3

		RSTN_[mode] = RSTN_[mode].cuda()
		RSTN_[mode].train()
		epoch = math.ceil(max_iterations[mode] / (batch_size * len(trainloader)))

		try: 
			for e in range(epoch):
				total_loss = 0.0
				start = time.time()
				# scheduler.step() 
				for index, (image, label) in enumerate(trainloader):
					start_it = time.time()        
					optimizer.zero_grad()
					image, label = image.cuda().float(), label.cuda().float()
					coarse_prob, fine_prob = RSTN_[mode](image, label)
					loss = COARSE_WEIGHT * criterion(coarse_prob, label) \
							+ (1 - COARSE_WEIGHT) * criterion(fine_prob, label)
					total_loss += loss.item()
					loss.backward()
					optimizer.step()
					print (plane + mode, "Epoch[%02d/%d], Iter[%04d], Train DSC %.4f Time Elapsed %.4fs" \
							%(e+1, epoch, index, 1 - loss.item(), time.time()-start_it)) 
				
				print (plane + mode, "Epoch[%02d/%d], Train Avg DSC: %.4f, Time elapsed %.2fs" \
						%(e+1, epoch, 1 - total_loss/len(trainloader), time.time()-start))

		except KeyboardInterrupt:
			print('!' * 10 , 'save before quitting ...')
		finally:
			snapshot_name = 'FD' + str(current_fold) + ':' + \
				plane + mode + str(slice_thickness) + '_' + str(organ_ID) + '_' + timestamp
			RSTN_snapshot[mode] = os.path.join(snapshot_path, snapshot_name) + '.pkl'
			torch.save(RSTN_[mode].state_dict(), RSTN_snapshot[mode])
			print('#' * 10 , 'end of ' + plane + mode + ' training stage!')

