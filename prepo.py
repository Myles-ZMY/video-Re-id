import os
import argparse
import numpy as np
from scipy.misc import imread
import h5py
import random
import json

def create_data(split, n1, n2, name, path, seq_length, img_index):
    data_class = split
    n_person = n1
    n_img = n2
    file_name = name
    file_path = path
    length = seq_length
    index = img_index
    number = np.zeros(n_person, dtype = 'uint32')
    img_data = np.zeros((n_img , 3, 128, 64), dtype = 'float32')
    for i,name in enumerate(file_name):
        no = int(name.split('_')[1])
        print('process the %dth person in %s data, it is the No.%d person' %(i+1, data_class, no))
        number[i] = no
        for j in range(length[i]):
            img = '%04d.png' %(j+1)
            img_path = os.path.join(file_path, name, img)
            x = imread(img_path)
            x = x.transpose(2, 0, 1)
            r, g, b = x[0], x[1], x[2]
            y = 0.299 * r + 0.587* g + 0.114 * b
            u = -0.1678 * r + -03313 * g + 0.5 * b 
            v = 0.5 * r + -0.4187 * g + -0.0813 * b
            x[0], x[1], x[2] = y, u, v
            img_data[j + index[i]] = x
        
    return number, img_data

def normalize(data1, data2, data3, data4):
    print('doing the normalization of raw data to have zero mean and unit variance')
    imgs = np.concatenate((data1, data2, data3, data4), axis = 0)
    imgs -= np.mean(imgs, axis = 0)
    imgs /= np.std(imgs, axis = 0)
    n1, n2, n3, n4 = data1.shape[0], data2.shape[0], data3.shape[0], data4.shape[0]
    data_split = np.split(imgs, [n1, n1 + n2, n1 + n2 + n3, n1 + n1 + n3 + n4])

    return data_split[0], data_split[1], data_split[2], data_split[3]
    

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


    for i in range(1, n_train):
        index1_train[i] = index1_train[i-1] + seq_len1_train[i-1]
        index2_train[i] = index2_train[i-1] + seq_len2_train[i-1]

    for i in range(1, n_test):
        index1_test[i] = index1_test[i-1] + seq_len1_test[i-1]
        index2_test[i] = index2_test[i-1] + seq_len2_test[i-1]
    
    no_train, imgs_data1_train = create_data('train', n_train, img_n1_train, file_name_train, file_path1, seq_len1_train, index1_train)
    no_train, imgs_data2_train = create_data('train', n_train, img_n2_train, file_name_train, file_path2, seq_len2_train, index2_train)
    no_test, imgs_data1_test = create_data('test', n_test, img_n1_test, file_name_test, file_path1, seq_len1_test, index1_test)
    no_test, imgs_data2_test = create_data('test', n_test, img_n2_test, file_name_test, file_path2, seq_len2_test, index2_test)
    
    imgs_data1_train, imgs_data2_train, imgs_data1_test, imgs_data2_test = normalize(imgs_data1_train, imgs_data2_train, imgs_data1_test, imgs_data2_test)

    index1_train = index1_train + 1
    index2_train = index2_train + 1 #in lua index start from 1
    index1_test = index1_test + 1
    index2_test = index2_test + 1

    f1 = h5py.File(os.path.join(params['file_root'],params['train_output']), 'w')
    f1.create_dataset('imgs1', dtype = 'float32', data = imgs_data1_train)
    f1.create_dataset('imgs2', dtype = 'float32', data = imgs_data2_train)
    f1.create_dataset('seq_length1', dtype = 'uint32', data = seq_len1_train)
    f1.create_dataset('seq_length2', dtype = 'uint32', data = seq_len2_train)
    f1.create_dataset('index1', dtype = 'uint32', data = index1_train)
    f1.create_dataset('index2', dtype = 'uint32', data = index2_train)
    f1.create_dataset('number', dtype = 'uint32', data = no_train)
    f1.close()
    print 'wrote', params['train_output']

    f2 = h5py.File(os.path.join(params['file_root'],params['test_output']), 'w')
    f2.create_dataset('imgs1', dtype = 'float32', data = imgs_data1_test)
    f2.create_dataset('imgs2', dtype = 'float32', data = imgs_data2_test)
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
    parser.add_argument('--person_num_train', default = 100, help = 'amount of people split to train data')
    parser.add_argument('--person_num_test', default = 100, help = 'amount of people split to test data')
    parser.add_argument('--train_output', default = 'data_train.h5', help = 'output file for training')
    parser.add_argument('--test_output', default = 'data_test.h5', help = 'output file for testing')


    args = parser.parse_args()
    params = vars(args)
    print 'parsed input parameters'
    print json.dumps(params, indent = 2)
    main(params)
