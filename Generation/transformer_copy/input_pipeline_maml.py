import time

import pickle
from tqdm import trange
import tensorflow as tf

print(time.strftime('%H:%M:%S', time.localtime(time.time())))

domains = [
    'Arts_Crafts_and_Sewing',
    'CDs_and_Vinyl',
    'Digital_Music',
    'Electronics',
    # 'Kindle_Store',
    'Movies_and_TV',
    'Video_Games',
    'Toys_and_Games',
]  # todo 7 注释掉目标域

d_path = []
for domain in domains:
    read_file_path = '../../datasets/8_dataset_final_mmd/dataset_final_mmd_toys/' + domain + '.pickle'  # todo 7 修改目标域文件夹
    # read_file_path = '../../datasets/5_dataset_final/' + domain + '.pickle'  # todo no_mmd+maml 修改目标域文件夹
    d_path.append(read_file_path)

encoder_max_len = 138
decoder_max_len = 77

support_set_batch_size = 7  # 一个batch里的support set数量
query_set_batch_size = 7  # 一个batch里的support set数量
domains_num = len(domains)  # domain个数：n_ways = 8-1
task_size = support_set_batch_size + query_set_batch_size  # 7
each_domain_for_task = int(task_size / domains_num)  # 每个域取的样本个数: 7/7=2
each_domain_size = 500  # 每个域的数据量
batch_nums = int(each_domain_size / each_domain_for_task)  # batch_nums = 500

print('-----MAML Input Pipeline Start!!-----')


# 生成数据, 参数为评论和摘要文本
def gen(dd_path):
    domain_contents = []
    for path in dd_path:  # 读取每个域的文件
        with open(path, 'rb') as f:
            domain_content = pickle.load(f)  # domain_content(list)包含一个域的100条数据
        # domain_content = domain_content[:500]
        domain_contents.append(domain_content)  # domain_contents(list)里面就有4个list，每个list是100条数据

    contents = []  # 存放所有的数据
    for i in range(batch_nums):  # 循环500次
        for domain in domain_contents:  # 遍历每个域，7个域
            for n in range(each_domain_for_task):  # 2
                if n % 2 == 0:
                    contents.insert(0, domain.pop())
                else:
                    contents.append(domain.pop())
    del domain_contents

    for _, content in zip(trange(len(contents)), contents):  # contents: 7000
        review_ids = tf.reshape(content['review_ids'], [encoder_max_len, ])
        embed_enc_input = tf.reshape(content['embed_enc_input'], [encoder_max_len, 512])
        extended_review_ids = content['extended_review_ids']
        review_oov_list = content['review_oov_list']
        summary_ids = tf.reshape(content['summary_ids'], [decoder_max_len - 1, ])
        embed_dec_input = tf.reshape(content['embed_dec_input'], [decoder_max_len - 1, 512])
        extended_summary_ids = content['extended_summary_ids']
        annotation_ids = tf.reshape(content['annotation_ids'], [encoder_max_len, ])
        embed_ann_input = tf.reshape(content['embed_ann_input'], [encoder_max_len, 512])
        extended_annotation_ids = content['extended_annotation_ids']
        annotation_oov_list = content['annotation_oov_list']
        embed_r_y_input = tf.reshape(content['embed_r_y_input'], [encoder_max_len, 512])

        output = {
            'review_ids': review_ids,
            'embed_enc_input': embed_enc_input,
            'extended_review_ids': extended_review_ids,
            'review_oov_list': review_oov_list,
            'summary_ids': summary_ids,
            'embed_dec_input': embed_dec_input,
            'extended_summary_ids': extended_summary_ids,
            'annotation_ids': annotation_ids,
            'embed_ann_input': embed_ann_input,
            'extended_annotation_ids': extended_annotation_ids,
            'annotation_oov_list': annotation_oov_list,
            'embed_r_y_input': embed_r_y_input
        }
        yield output


# from_generator
print(time.strftime('%H:%M:%S', time.localtime(time.time())))
print('Generating dataset...')
dataset = tf.data.Dataset.from_generator(generator=gen,
                                         args=[d_path],
                                         output_signature={
                                             'review_ids': tf.TensorSpec(shape=(encoder_max_len,), dtype=tf.int32),
                                             'embed_enc_input': tf.TensorSpec(shape=(encoder_max_len, 512),
                                                                              dtype=tf.float32),
                                             'extended_review_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                                             'review_oov_list': tf.TensorSpec(shape=(None,), dtype=tf.string),
                                             'summary_ids': tf.TensorSpec(shape=(decoder_max_len - 1,),
                                                                          dtype=tf.int32),
                                             'embed_dec_input': tf.TensorSpec(shape=(decoder_max_len - 1, 512),
                                                                              dtype=tf.float32),
                                             'extended_summary_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                                             'annotation_ids': tf.TensorSpec(shape=(encoder_max_len,),
                                                                             dtype=tf.int32),
                                             'embed_ann_input': tf.TensorSpec(shape=(encoder_max_len, 512),
                                                                              dtype=tf.float32),
                                             'extended_annotation_ids': tf.TensorSpec(shape=(None,),
                                                                                      dtype=tf.int32),
                                             'annotation_oov_list': tf.TensorSpec(shape=(None,), dtype=tf.string),
                                             'embed_r_y_input': tf.TensorSpec(shape=(encoder_max_len, 512),
                                                                              dtype=tf.float32)
                                         })

# 取batch同时对齐 padded_batch
dataset = dataset.padded_batch(batch_size=task_size,
                               padded_shapes=({
                                   'review_ids': (encoder_max_len,),
                                   'embed_enc_input': (encoder_max_len, 512),
                                   'extended_review_ids': (encoder_max_len,),
                                   'review_oov_list': (None,),
                                   'summary_ids': (decoder_max_len - 1,),
                                   'embed_dec_input': (decoder_max_len - 1, 512),
                                   'extended_summary_ids': (decoder_max_len - 1,),
                                   'annotation_ids': (encoder_max_len,),
                                   'embed_ann_input': (encoder_max_len, 512),
                                   'extended_annotation_ids': (encoder_max_len,),
                                   'annotation_oov_list': (None,),
                                   'embed_r_y_input': (encoder_max_len, 512)
                               }),
                               padding_values=({
                                   'review_ids': 0,
                                   'embed_enc_input': 0.0,
                                   'extended_review_ids': 0,
                                   'review_oov_list': b'',
                                   'summary_ids': 0,
                                   'embed_dec_input': 0.0,
                                   'extended_summary_ids': 0,
                                   'annotation_ids': 0,
                                   'embed_ann_input': 0.0,
                                   'extended_annotation_ids': 0,
                                   'annotation_oov_list': b'',
                                   'embed_r_y_input': 0.0
                               }),
                               drop_remainder=True)

dataset_maml = []
for _, batch_content in enumerate(dataset):
    dataset_maml.append(batch_content)

print('The batch size of MAML dataset:', task_size)
print('The number of batch in MAML dataset:', len(dataset_maml))
print(time.strftime('%H:%M:%S', time.localtime(time.time())))
print('-----MAML Input_pipeline End-----')
