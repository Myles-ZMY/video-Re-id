import os
import argparse
import numpy as np
from scipy.misc import imread
import h5py
import json

def main(params):
    file_path1 = os.path.join(params['file_root'], 'camera_a')
    file_path2 = os.path.join(params['file_root'], 'camera_b')
    seq_len1 = np.zeros(200, dtype = 'uint32')
    seq_len2 = np.zeros(200, dtype = 'uint32')
    index1 = np.zeros(200, dtype = 'uint32')
    index2 = np.zeros(200, dtype = 'uint32')
    index1[0] = 1 # in torch/lua index start from 1
    index2[0] = 1
    
    file_name1 = os.listdir(file_path1)

    for i,name in enumerate(file_name1):
        imgs1 = os.listdir(os.path.join(file_path1,name))
        imgs2 = os.listdir(os.path.join(file_path2,name))
        seq_len1[i] = min(len(imgs1), params['max_length'])
        seq_len2[i] = min(len(imgs2), params['max_length'])

    img_n1 = np.sum(seq_len1)
    img_n2 = np.sum(seq_len2)

    imgs_data1 = np.zeros((img_n1, 3, 128, 64), dtype = 'uint8')
    imgs_data2 = np.zeros((img_n2, 3, 128, 64), dtype = 'uint8')

    for i,name in enumerate(file_name1):
        imgs1 = os.listdir(os.path.join(file_path1, name))
        imgs2 = os.listdir(os.path.join(file_path2, name))
        for j,img in enumerate(imgs1):
            img_path = os.path.join(file_path1, name, img)
            x = imread(img_path)
            x = x.transpose(2, 0, 1)
            imgs_data1[j] = x
        for j,img in enumerate(imgs2):
            img_path = os.path.join(file_path2, name, img)
            x = imread(img_path)
            x = x.transpose(2, 0, 1)
            imgs_data2[j] = x
            
    for i in range(1, 200):
        index1[i] = index1[i-1] + seq_len1[i-1]
        index2[i] = index2[i-1] + seq_len2[i-2]

    f1 = h5py.File(os.path.join(params['file_root'],params['output1']), 'w')
    f1.create_dataset('imgs', dtype = 'uint8', data = imgs_data1)
    f1.create_dataset('seq_length', dtype = 'uint32', data = seq_len1)
    f1.create_dataset('index', dtype = 'uint32', data = index1)
    f1.close()
    print 'wrote', params['output1']

    f2 = h5py.File(os.path.join(params['file_root'],params['output2']), 'w')
    f2.create_dataset('imgs', dtype = 'uint8', data = imgs_data2)
    f2.create_dataset('seq_length', dtype = 'uint32', data = seq_len2)
    f2.create_dataset('index', dtype = 'uint32', data = index2)
    f2.close()
    print 'wrote', params['output2']
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_root', required = True, help = 'input image files root path')
    parser.add_argument('--max_length', default = 192, type = int, help = 'max length of video frame')
    parser.add_argument('--output1', default = 'data1.h5', help = 'output file from camera a')
    parser.add_argument('--output2', default = 'data2.h5', help = 'output file from camera b')

    args = parser.parse_args()
    params = vars(args)
    print 'parsed input parameters'
    print json.dumps(params, indent = 2)
    main(params)
