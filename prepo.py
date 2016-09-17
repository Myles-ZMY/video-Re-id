import os
import argparse
import numpy as np
from scipy.misc import imread
import h5py
import random
import json

def main(params):
    file_path1 = os.path.join(params['file_root'], 'camera_a')
    file_path2 = os.path.join(params['file_root'], 'camera_b')

    file_name = os.listdir(file_path1)
    random.shuffle(file_name)

    n_person = len(file_name)
    n_train = params['person_num_train']
    n_test = params['person_num_test']
    assert n_person == (n_train + n_test), 'total people number should be equal to the sum of train and test people'
    seq_len1_train = np.zeros(n_train, dtype = 'uint32')
    seq_len2_train = np.zeros(n_train, dtype = 'uint32')
    seq_len1_test = np.zeros(n_test, dtype = 'uint32')
    seq_len2_test = np.zeros(n_test, dtype = 'uint32')
    index1_train = np.zeros(n_train, dtype = 'uint32')
    index2_train = np.zeros(n_train, dtype = 'uint32')
    index1_test = np.zeros(n_test, dtype = 'uint32')
    index2_test = np.zeros(n_test, dtype = 'uint32')
    no_train = np.zeros(n_train, dtype = 'uint32')
    no_test = np.zeros(n_test, dtype = 'uint32')
    index1_train[0] = 0
    index2_train[0] = 0
    index1_test[0] = 0
    index2_test[0] = 0

    print('There are %d persons split for training, and %d persons split for testing.There are %d persons totally' %(n_train, n_test, n_person))

    file_name_train = []
    file_name_test = []
    for i,name in enumerate(file_name):
        if i < n_train:
            file_name_train.append(name)
        else:
            file_name_test.append(name)

    for i, name in enumerate(file_name_train):
        imgs1 = os.listdir(os.path.join(file_path1, name))
        imgs2 = os.listdir(os.path.join(file_path2, name))
        seq_len1_train[i] = min(len(imgs1), params['max_length'])
        seq_len2_train[i] = min(len(imgs2), params['max_length'])

    for i, name in enumerate(file_name_test):
        imgs1 = os.listdir(os.path.join(file_path1, name))
        imgs2 = os.listdir(os.path.join(file_path2, name))
        seq_len1_test[i] = min(len(imgs1), params['max_length'])
        seq_len2_test[i] = min(len(imgs2), params['max_length'])

    img_n1_train = np.sum(seq_len1_train)
    img_n2_train = np.sum(seq_len2_train)
    print('There are %d images in probe camera and %d images in gallery camera for training' %(img_n1_train, img_n2_train))
    img_n1_test = np.sum(seq_len1_test)
    img_n2_test = np.sum(seq_len2_test)
    print('There are %d images in probe camera and %d images in gallery camera for testing' %(img_n1_test, img_n2_test))

    imgs_data1_train = np.zeros((img_n1_train, 3, 128, 64), dtype = 'uint8')
    imgs_data2_train = np.zeros((img_n2_train, 3, 128, 64), dtype = 'uint8')
    imgs_data1_test = np.zeros((img_n1_test, 3, 128, 64), dtype = 'uint8')
    imgs_data2_test = np.zeros((img_n2_test, 3, 128, 64), dtype = 'uint8')

    for i in range(1, n_train):
        index1_train[i] = index1_train[i-1] + seq_len1_train[i-1]
        index2_train[i] = index2_train[i-1] + seq_len2_train[i-1]

    for i in range(1, n_test):
        index1_test[i] = index1_test[i-1] + seq_len1_test[i-1]
        index2_test[i] = index2_test[i-1] + seq_len2_test[i-1]


    for i,name in enumerate(file_name_train):
        no = int(name.split('_')[1])
        print('processing the %dth person in train data, it is the NO.%d person' %(i+1, no))
        no_train[i] = no
        for j in range(seq_len1_train[i]):
            img = '%04d.png' %(j+1)
            img_path = os.path.join(file_path1, name, img)
            x = imread(img_path)
            x = x.transpose(2, 0, 1)
            imgs_data1_train[j+index1_train[i]] = x
        for j in range(seq_len2_train[i]):
            img = '%04d.png' %(j+1)
            img_path = os.path.join(file_path2, name, img)
            x = imread(img_path)
            x = x.transpose(2, 0, 1)
            imgs_data2_train[j+index2_train[i]] = x

    for i,name in enumerate(file_name_test):
        no = int(name.split('_')[1])
        print('processing the %dth person in test data, it is the NO.%d person' %(i+1, no))
        no_test[i] = no
        for j in range(seq_len1_test[i]):
            img = '%04d.png' %(j+1)
            img_path = os.path.join(file_path1, name, img)
            x = imread(img_path)
            x = x.transpose(2, 0, 1)
            imgs_data1_test[j+index1_test[i]] = x
        for j in range(seq_len2_test[i]):
            img = '%04d.png' %(j+1)
            img_path = os.path.join(file_path2, name, img)
            x = imread(img_path)
            x = x.transpose(2, 0, 1)
            imgs_data2_test[j+index2_test[i]] = x

    index1_train = index1_train + 1
    index2_train = index2_train + 1 #in lua index start from 1
    index1_test = index1_test + 1
    index2_test = index2_test + 1

    f1 = h5py.File(os.path.join(params['file_root'],params['train_output']), 'w')
    f1.create_dataset('imgs1', dtype = 'uint8', data = imgs_data1_train)
    f1.create_dataset('imgs2', dtype = 'uint8', data = imgs_data2_train)
    f1.create_dataset('seq_length1', dtype = 'uint32', data = seq_len1_train)
    f1.create_dataset('seq_length2', dtype = 'uint32', data = seq_len2_train)
    f1.create_dataset('index1', dtype = 'uint32', data = index1_train)
    f1.create_dataset('index2', dtype = 'uint32', data = index2_train)
    f1.create_dataset('number', dtype = 'uint32', data = no_train)
    f1.close()
    print 'wrote', params['train_output']

    f2 = h5py.File(os.path.join(params['file_root'],params['test_output']), 'w')
    f2.create_dataset('imgs1', dtype = 'uint8', data = imgs_data1_test)
    f2.create_dataset('imgs2', dtype = 'uint8', data = imgs_data2_test)
    f2.create_dataset('seq_length1', dtype = 'uint32', data = seq_len1_test)
    f2.create_dataset('seq_length2', dtype = 'uint32', data = seq_len2_test)
    f2.create_dataset('index1', dtype = 'uint32', data = index1_test)
    f2.create_dataset('index2', dtype = 'uint32', data = index2_test)
    f2.create_dataset('number', dtype = 'uint32', data = no_test)
    f2.close()
    print 'wrote', params['test_output']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_root', default = '.', help = 'input image files root path')
    parser.add_argument('--max_length', default = 192, type = int, help = 'max length of video frame')
    parser.add_argument('--person_num_train', default = 150, help = 'amount of people split to train data')
    parser.add_argument('--person_num_test', default = 50, help = 'amount of people split to test data')
    parser.add_argument('--train_output', default = 'data_train.h5', help = 'output file for training')
    parser.add_argument('--test_output', default = 'data_test.h5', help = 'output file for testing')


    args = parser.parse_args()
    params = vars(args)
    print 'parsed input parameters'
    print json.dumps(params, indent = 2)
    main(params)
