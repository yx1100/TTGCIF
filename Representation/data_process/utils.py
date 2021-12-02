import pickle
import tensorflow as tf


def combine_file(domain):
    """
    合并2个4k数据集(legacy)
    :param domain:
    :return:
    """
    with open('./datasets/dataset_final/' + domain + '1.pickle', 'rb') as f:
        p1 = pickle.load(f)

    with open('./datasets/dataset_final/' + domain + '2.pickle', 'rb') as ff:
        p2 = pickle.load(ff)

    if p1[0]['extended_review_ids'] != p2[0]['extended_review_ids']:
        print(p1[0]['review_ids'])
        print(p2[0]['review_ids'])
        p = p1 + p2
        with open('./datasets/dataset_final/' + domain + '.pickle', 'wb') as fff:
            pickle.dump(p, fff)
        print('finish')
    else:
        print('stop')


def check_data_shape(path):
    """
    # 检查最终生成的数据的正确性
    :param path:
    :return:
    """
    with open(path, 'rb') as f:
        p = pickle.load(f)

    # print(type(p)) -> list
    # print(type(p[0])) -> dict

    i = 0
    exit_flag = False
    a1 = tf.zeros([1, 162])
    a2 = tf.zeros([1, 162, 128])
    a3 = tf.zeros([1, 118])
    a4 = tf.zeros([1, 118, 128])
    for n in p:
        i = i + 1
        for key, value in n.items():
            if key == 'review_ids' or key == 'annotation_ids':
                if value.shape != a1.shape:
                    print('error1')
                    exit_flag = True
                    break
            if key == 'embed_enc_input' or key == 'embed_ann_input' or key == 'embed_r_y_input' or key == 'embed_enc_mmd_input':
                if value.shape != a2.shape:
                    print('error2')
                    exit_flag = True
                    break
            if key == 'summary_ids':
                if value.shape != a3.shape:
                    exit_flag = True
                    print('error3')
                    break
            if key == 'embed_dec_input':
                if value.shape != a4.shape:
                    print('error4')
                    exit_flag = True
                    break
            if key == 'review_oov_list' or key == 'annotation_oov_list':  # 检查oov list
                if len(value) != 0:
                    print('error5')
                    exit_flag = True
                    break
            if key == 'extended_review_ids' or key == 'extended_summary_ids' or key == 'extended_annotation_ids':
                if len(value) == 0:
                    print('error6')
                    exit_flag = True
                    break

            # if isinstance(value, list):
            #     print(key, ' list ', len(value))
            # else:
            #     print(key, ' ', value.shape)

        if exit_flag:
            break
        print(i)


# combine_file('Video_Games')
domains = [
    'Arts_Crafts_and_Sewing',
    'CDs_and_Vinyl',
    'Digital_Music',
    'Electronics',
    'Kindle_Store',
    'Movies_and_TV',
    'Toys_and_Games',
    'Video_Games'
]
for d in domains:
    check_data_shape('/Users/yuxin/Developer/TPCC/datasets/5_dataset_final/' + d + '.pickle')
