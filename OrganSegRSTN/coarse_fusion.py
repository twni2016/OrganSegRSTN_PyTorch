import numpy as np
import os
import sys
import time
from utils import *


data_path = sys.argv[1]
current_fold = int(sys.argv[2])
organ_number = int(sys.argv[3])
low_range = int(sys.argv[4])
high_range = int(sys.argv[5])
slice_threshold = float(sys.argv[6])
slice_thickness = int(sys.argv[7])
organ_ID = int(sys.argv[8])
GPU_ID = int(sys.argv[9])
learning_rate1 = float(sys.argv[10])
learning_rate_m1 = int(sys.argv[11])
learning_rate2 = float(sys.argv[12])
learning_rate_m2 = int(sys.argv[13])
crop_margin = int(sys.argv[14])
result_path = os.path.join(result_path, 'coarse_testing_' + \
	sys.argv[10] + 'x' + str(learning_rate_m1) + ',' + str(crop_margin))
epoch = 'e' + sys.argv[15] + sys.argv[16] + sys.argv[17] + sys.argv[18]
epoch_list = [epoch]
threshold = float(sys.argv[19])
timestamp = {}
timestamp['X'] = sys.argv[20]
timestamp['Y'] = sys.argv[21]
timestamp['Z'] = sys.argv[22]

volume_list = open(testing_set_filename(current_fold), 'r').read().splitlines()
while volume_list[len(volume_list) - 1] == '':
    volume_list.pop()

result_name_ = {}
result_directory_ = {}
for plane in ['X', 'Y', 'Z']:
	result_name__ = result_name_from_timestamp(result_path, current_fold, \
		plane, 'J', slice_thickness, organ_ID, volume_list, timestamp[plane])
	if result_name__ == '':
		exit('  Error: no valid result directories are detected!')
	result_directory__ = os.path.join(result_path, result_name__, 'volumes')
	print('  Result directory for plane ' + plane + ': ' + result_directory__ + ' .')
	if result_name__.startswith('FD'):
		index_ = result_name__.find(':')
		result_name__ = result_name__[index_ + 1: ]
	result_name_[plane] = result_name__
	result_directory_[plane] = result_directory__

DSC_X = np.zeros((len(volume_list)))
DSC_Y = np.zeros((len(volume_list)))
DSC_Z = np.zeros((len(volume_list)))
DSC_F1 = np.zeros((len(volume_list)))
DSC_F2 = np.zeros((len(volume_list)))
DSC_F3 = np.zeros((len(volume_list)))
DSC_F1P = np.zeros((len(volume_list)))
DSC_F2P = np.zeros((len(volume_list)))
DSC_F3P = np.zeros((len(volume_list)))

result_name = 'FD' + str(current_fold) + ':' + 'fusion:' + result_name_['X'] + ',' + \
    result_name_['Y'] + ',' + result_name_['Z'] + '_' + epoch + ',' + str(threshold)
result_directory = os.path.join(result_path, result_name, 'volumes')
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

result_file = os.path.join(result_path, result_name, 'results.txt')
output = open(result_file, 'w')
output.close()
output = open(result_file, 'a+')
output.write('Fusing results of ' + str(len(epoch_list)) + ' snapshots:\n')
output.close()

for i in range(len(volume_list)):
    start_time = time.time()
    print('Testing ' + str(i + 1) + ' out of ' + str(len(volume_list)) + ' testcases.')
    output = open(result_file, 'a+')
    output.write('  Testcase ' + str(i + 1) + ':\n')
    output.close()
    s = volume_list[i].split(' ')
    label = np.load(s[2])
    label = is_organ(label, organ_ID).astype(np.uint8)

    for plane in ['X', 'Y', 'Z']:
        volume_file = volume_filename_fusion(result_directory, plane, i)
        pred = np.zeros(label.shape, dtype = np.float32)
        for t in range(len(epoch_list)):
            volume_file_ = volume_filename_testing(result_directory_[plane], epoch_list[t], i)
            pred += np.load(volume_file_)['volume']
        pred_ = (pred >= threshold * 255 * len(epoch_list))
        if not os.path.isfile(volume_file):
            np.savez_compressed(volume_file, volume = pred_)
        DSC_, inter_sum, pred_sum, label_sum = DSC_computation(label, pred_)
        print('    DSC_' + plane + ' = 2 * ' + str(inter_sum) + ' / (' + \
            str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_) + ' .')
        output = open(result_file, 'a+')
        output.write('    DSC_' + plane + ' = 2 * ' + str(inter_sum) + ' / (' + \
            str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_) + ' .\n')
        output.close()
        if pred_sum == 0 and label_sum == 0:
            DSC_ = 0
        pred /= (255 * len(epoch_list))
        if plane == 'X':
            pred_X = pred
            DSC_X[i] = DSC_
        elif plane == 'Y':
            pred_Y = pred
            DSC_Y[i] = DSC_
        elif plane == 'Z':
            pred_Z = pred
            DSC_Z[i] = DSC_

    volume_file_F1 = volume_filename_fusion(result_directory, 'F1', i)
    volume_file_F2 = volume_filename_fusion(result_directory, 'F2', i)
    volume_file_F3 = volume_filename_fusion(result_directory, 'F3', i)

    if not os.path.isfile(volume_file_F1) or not os.path.isfile(volume_file_F2) or \
        not os.path.isfile(volume_file_F3):
        pred_total = pred_X + pred_Y + pred_Z
    if os.path.isfile(volume_file_F1):
        pred_F1 = np.load(volume_file_F1)['volume'].astype(np.uint8)
    else:
        pred_F1 = (pred_total >= 0.5).astype(np.uint8)
        np.savez_compressed(volume_file_F1, volume = pred_F1)
    DSC_F1[i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred_F1)
    print('    DSC_F1 = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' \
        + str(label_sum) + ') = ' + str(DSC_F1[i]) + ' .')
    output = open(result_file, 'a+')
    output.write('    DSC_F1 = 2 * ' + str(inter_sum) + ' / (' + \
        str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_F1[i]) + ' .\n')
    output.close()
    if pred_sum == 0 and label_sum == 0:
        DSC_F1[i] = 0

    if os.path.isfile(volume_file_F2):
        pred_F2 = np.load(volume_file_F2)['volume'].astype(np.uint8)
    else:
        pred_F2 = (pred_total >= 1.5).astype(np.uint8)
        np.savez_compressed(volume_file_F2, volume = pred_F2)
    DSC_F2[i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred_F2)
    print('    DSC_F2 = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' + \
        str(label_sum) + ') = ' + str(DSC_F2[i]) + ' .')
    output = open(result_file, 'a+')
    output.write('    DSC_F2 = 2 * ' + str(inter_sum) + ' / (' + \
        str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_F2[i]) + ' .\n')
    output.close()
    if pred_sum == 0 and label_sum == 0:
        DSC_F2[i] = 0
    
    if os.path.isfile(volume_file_F3):
        pred_F3 = np.load(volume_file_F3)['volume'].astype(np.uint8)
    else:
        pred_F3 = (pred_total >= 2.5).astype(np.uint8)
        np.savez_compressed(volume_file_F3, volume = pred_F3)
    DSC_F3[i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred_F3)
    print('    DSC_F3 = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' + \
        str(label_sum) + ') = ' + str(DSC_F3[i]) + ' .')
    output = open(result_file, 'a+')
    output.write('    DSC_F3 = 2 * ' + str(inter_sum) + ' / (' + \
        str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_F3[i]) + ' .\n')
    output.close()
    if pred_sum == 0 and label_sum == 0:
        DSC_F3[i] = 0

    volume_file_F1P = volume_filename_fusion(result_directory, 'F1P', i)
    volume_file_F2P = volume_filename_fusion(result_directory, 'F2P', i)
    volume_file_F3P = volume_filename_fusion(result_directory, 'F3P', i)
    S = pred_F3
    if (S.sum() == 0):
        S = pred_F2
    if (S.sum() == 0):
        S = pred_F1

    if os.path.isfile(volume_file_F1P):
        pred_F1P = np.load(volume_file_F1P)['volume'].astype(np.uint8)
    else:
        pred_F1P = post_processing(pred_F1, S, 0.5, organ_ID)
        np.savez_compressed(volume_file_F1P, volume = pred_F1P)
    DSC_F1P[i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred_F1P)
    print('    DSC_F1P = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' + \
        str(label_sum) + ') = ' + str(DSC_F1P[i]) + ' .')
    output = open(result_file, 'a+')
    output.write('    DSC_F1P = 2 * ' + str(inter_sum) + ' / (' + \
        str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_F1P[i]) + ' .\n')
    output.close()
    if pred_sum == 0 and label_sum == 0:
        DSC_F1P[i] = 0

    if os.path.isfile(volume_file_F2P):
        pred_F2P = np.load(volume_file_F2P)['volume'].astype(np.uint8)
    else:
        pred_F2P = post_processing(pred_F2, S, 0.5, organ_ID)
        np.savez_compressed(volume_file_F2P, volume = pred_F2P)
    DSC_F2P[i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred_F2P)
    print('    DSC_F2P = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' + \
        str(label_sum) + ') = ' + str(DSC_F2P[i]) + ' .')
    output = open(result_file, 'a+')
    output.write('    DSC_F2P = 2 * ' + str(inter_sum) + ' / (' + \
        str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_F2P[i]) + ' .\n')
    output.close()
    if pred_sum == 0 and label_sum == 0:
        DSC_F2P[i] = 0

    if os.path.isfile(volume_file_F3P):
        pred_F3P = np.load(volume_file_F3P)['volume'].astype(np.uint8)
    else:
        pred_F3P = post_processing(pred_F3, S, 0.5, organ_ID)
        np.savez_compressed(volume_file_F3P, volume = pred_F3P)
    DSC_F3P[i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred_F3P)
    print('    DSC_F3P = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' + \
        str(label_sum) + ') = ' + str(DSC_F3P[i]) + ' .')
    output = open(result_file, 'a+')
    output.write('    DSC_F3P = 2 * ' + str(inter_sum) + ' / (' + \
        str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_F3P[i]) + ' .\n')
    output.close()
    if pred_sum == 0 and label_sum == 0:
        DSC_F3P[i] = 0

    pred_X = None
    pred_Y = None
    pred_Z = None
    pred_F1 = None
    pred_F2 = None
    pred_F3 = None
    pred_F1P = None
    pred_F2P = None
    pred_F3P = None

output = open(result_file, 'a+')
print('Average DSC_X = ' + str(np.mean(DSC_X)) + ' .')
output.write('Average DSC_X = ' + str(np.mean(DSC_X)) + ' .\n')
print('Average DSC_Y = ' + str(np.mean(DSC_Y)) + ' .')
output.write('Average DSC_Y = ' + str(np.mean(DSC_Y)) + ' .\n')
print('Average DSC_Z = ' + str(np.mean(DSC_Z)) + ' .')
output.write('Average DSC_Z = ' + str(np.mean(DSC_Z)) + ' .\n')
print('Average DSC_F1 = ' + str(np.mean(DSC_F1)) + ' .')
output.write('Average DSC_F1 = ' + str(np.mean(DSC_F1)) + ' .\n')
print('Average DSC_F2 = ' + str(np.mean(DSC_F2)) + ' .')
output.write('Average DSC_F2 = ' + str(np.mean(DSC_F2)) + ' .\n')
print('Average DSC_F3 = ' + str(np.mean(DSC_F3)) + ' .')
output.write('Average DSC_F3 = ' + str(np.mean(DSC_F3)) + ' .\n')
print('Average DSC_F1P = ' + str(np.mean(DSC_F1P)) + ' .')
output.write('Average DSC_F1P = ' + str(np.mean(DSC_F1P)) + ' .\n')
print('Average DSC_F2P = ' + str(np.mean(DSC_F2P)) + ' .')
output.write('Average DSC_F2P = ' + str(np.mean(DSC_F2P)) + ' .\n')
print('Average DSC_F3P = ' + str(np.mean(DSC_F3P)) + ' .')
output.write('Average DSC_F3P = ' + str(np.mean(DSC_F3P)) + ' .\n')
output.close()
print('The fusion process is finished.')