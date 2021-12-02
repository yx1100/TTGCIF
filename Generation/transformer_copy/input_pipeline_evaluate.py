import pickle
import tensorflow as tf
import math

"""
数据对接改动
"""

encoder_max_len = 138
decoder_max_len = 77

batch_size = 1
fine_tune_rate = 0.1  # todo 7 这里和fine-tune时要同步

print('-----Evaluate Input_pipeline Start-----')

target_domain = 'Kindle_Store'  # todo 5 修改目标域

# read_file_path = '../../datasets/8_dataset_final_mmd/dataset_final_mmd_toys/' + target_domain + '.pickle'  # todo 5 修改目标域文件夹
read_file_path = '../../datasets/5_dataset_final/' + target_domain + '.pickle'  # todo no_mmd+zero-shot 修改目标域文件夹


# 生成数据, 参数为评论和摘要文本
def gen(d_path):
    with open(d_path, 'rb') as f:
        contents = pickle.load(f)

    fine_tune_size = math.floor(len(contents) * fine_tune_rate)
    contents = contents[fine_tune_size:]  # todo 7 这里涉及fine-tune，2阶段要注释掉；todo 8 3阶段要取消注释

    for content in contents:
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


dataset = tf.data.Dataset.from_generator(generator=gen,
                                         args=[read_file_path],
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
dataset = dataset.padded_batch(batch_size=batch_size,
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

dataset_evaluate = []
for _, batch_content in enumerate(dataset):
    dataset_evaluate.append(batch_content)

print('The batch size of Evaluate dataset:', batch_size)
print('The number of batch in Evaluate dataset:', len(dataset_evaluate))

print('-----Evaluate Input Pipeline End-----')
