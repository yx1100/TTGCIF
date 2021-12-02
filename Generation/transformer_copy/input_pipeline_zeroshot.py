import time

import pickle
from tqdm import trange
import tensorflow as tf

print(time.strftime('%H:%M:%S', time.localtime(time.time())))

domains = [
    # 'Arts_Crafts_and_Sewing',
    # 'CDs_and_Vinyl',
    # 'Digital_Music',
    # 'Electronics',
    # 'Movies_and_TV',
    # 'Video_Games',
    # 'Toys_and_Games',
    'Kindle_Store',
    # todo 4 将目标域移到最后一个
]
d_path = []
for domain in domains:
    # read_file_path = '../../datasets/8_dataset_final_mmd/dataset_final_mmd_arts/' + domain + '.pickle'  # todo 4 修改目标域文件夹
    read_file_path = '../../datasets/5_dataset_final/' + domain + '.pickle'  # todo no_mmd+zero-shot 修改目标域文件夹
    d_path.append(read_file_path)

encoder_max_len = 138
decoder_max_len = 77

BATCH_SIZE_OF_ZEROSHOT = 32

print('-----Zero-shot Learning Input Pipeline Start!!!-----')


# 生成数据, 参数为评论和摘要文本
def gen(d_path):
    contents = []
    for domain in d_path:
        with open(domain, 'rb') as f:
            cc = pickle.load(f)

        for c in cc:
            contents.append(c)

    for _, content in zip(trange(len(contents)), contents):
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
dataset = dataset.padded_batch(batch_size=BATCH_SIZE_OF_ZEROSHOT,
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

dataset_zeroshot = []
for _, batch_content in enumerate(dataset):
    dataset_zeroshot.append(batch_content)

print('The batch size of Zero-shot Learning dataset:', BATCH_SIZE_OF_ZEROSHOT)
print('The number of batch in Zero-shot Learning dataset:', len(dataset_zeroshot))

print(time.strftime('%H:%M:%S', time.localtime(time.time())))
print('-----Zero-shot Learning Input_pipeline End-----')
