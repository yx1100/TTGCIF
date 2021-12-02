import pickle

import numpy as np
import tensorflow as tf

domains1 = [
    'Arts_Crafts_and_Sewing',
    'CDs_and_Vinyl',
    'Digital_Music',
    'Electronics',
    'Kindle_Store',
    'Movies_and_TV',
    'Toys_and_Games',
    'Video_Games'
]
domains2 = [
    'Arts',
    'CDs',
    'Digital',
    'Electronics',
    'Kindle',
    'Movies',
    'Toys',
    'Video'
]
target = domains2[7]  # todo 3 修改目标域，数字+1
# todo 3 记得换回tf2的环境
for domain1, domain2 in zip(domains1, domains2):
    read_path1 = '../../datasets/7_dataset_mmd/' + target + '/' + domain2 + '_' + target + '_MMD.npy'
    read_path2 = '../../datasets/5_dataset_final/' + domain1 + '.pickle'
    save_path = '../../datasets/8_dataset_final_mmd/dataset_final_mmd_video/' + domain1 + '.pickle'  # todo 3 修改目标域文件夹

    data1 = np.load(read_path1)
    print(data1.shape[0])

    with open(read_path2, 'rb') as f:
        data2 = pickle.load(f)

    data2 = data2[:data1.shape[0]]
    print(len(data2))

    data = []
    for i in range(data1.shape[0]):
        data2[i]['embed_enc_input'] = tf.convert_to_tensor(np.reshape(data1[i], newshape=(1, 138, 512)))
        print(i)
        data.append(data2[i])

    print('len: ', len(data), '\n')
    print(data[199])

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
