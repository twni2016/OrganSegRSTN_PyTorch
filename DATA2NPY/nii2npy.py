import numpy as np
import os
import nibabel


N = 82
W = 512
H = 512
path1 = 'TCIA_pancreas_labels-02-05-2017'
path2 = 'labels'
if not os.path.exists(path2):
    os.makedirs(path2)

for n in range(N):
    volumeID = '{:0>4}'.format(n + 1)
    print 'Processing File ' + volumeID
    filename1 = 'label' + volumeID + '.nii.gz'
    directory1 = os.path.join(path1, filename1)
    filename2 = volumeID + '.npy'
    file1 = os.path.join(path1, filename1)
    data = nibabel.load(file1).get_data().transpose(1, 0, 2)
    print '  Data shape is ' + str(data.shape) + ' .'
    file2 = os.path.join(path2, filename2)
    np.save(file2, data)
    print 'File ' + volumeID + ' is saved in ' + file2 + ' .'

