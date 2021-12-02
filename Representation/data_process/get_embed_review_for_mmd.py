import pickle
import sys

import numpy as np
from tqdm import trange

"""
读取每条数据的embed review，保存为numpy文件，给mmd处理
"""

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

for domain1, domain2 in zip(domains1, domains2):
    read_file_path = '../../datasets/5_dataset_final/' + domain1 + '.pickle'
    save_file_path = '../../datasets/6_dataset_embed_review/' + domain2 + '.npy'

    print('当前处理文件名：', read_file_path + '\n')
    with open(read_file_path, 'rb') as f:
        contents = pickle.load(f)

    print('文本数量：', len(contents))

    temp_list = []
    for _, content in zip(trange(len(contents)), contents):
        embed_enc_input = content['embed_enc_input']
        embed_enc_input = embed_enc_input.numpy()
        embed_enc_input = np.reshape(embed_enc_input, newshape=(138, 512))
        temp_list.append(embed_enc_input)

    np.save(save_file_path, temp_list)
    print('-----finish-----')
