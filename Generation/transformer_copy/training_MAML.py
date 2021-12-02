# coding=utf-8
import time
import tensorflow as tf

from datetime import datetime
from transformers import BertTokenizer

from model.transformer import Transformer
from module.optimizer import CustomSchedule
from module.loss_and_metrics import train_loss, train_accuracy
from train_step import train_step_maml, train_step_fine_tune
from input_pipeline_maml import dataset_maml, support_set_batch_size, query_set_batch_size
from input_pipeline_finetune import dataset_fine_tune
from input_pipeline_evaluate import dataset_evaluate
from evaluate import for_evaluate

vocab_size = BertTokenizer.from_pretrained('./bert_uncased_L-8_H-512_A-8').vocab_size

num_layers = 4  # 6
d_model = 512
dff = 512  # 2048
num_heads = 8  # 8
dropout_rate = 0.1  # 0.1
enc_units = 512  # GRU
adapter_size = 128  # Adapter 128
inner_step = 1  # MAML

GLOBAL_MAX = 0  # todo: 修改每个目标域的初始最优值
PARAMETER_INIT = False
TOTAL_EPOCH = 100

print('-----开始训练-----')

# -----Optimizer-----
lr_shrink_factor = 1000
assert lr_shrink_factor == 1000
learning_rate = CustomSchedule(d_model, shrink_factor=lr_shrink_factor)
maml_inner_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)  # lr改动
maml_outer_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)  # lr改动

# 构建Transformer模型
transformer = Transformer(num_layers=num_layers,
                          d_model=d_model,
                          num_heads=num_heads,
                          dff=dff,
                          enc_units=enc_units,  # GRU
                          vocab_size=vocab_size,
                          rate=dropout_rate,
                          adapter_size=adapter_size)  # adapter

# -----TensorBoard-----
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)


def train_for_maml(dataset, dataset2, support_set_batch_size, query_set_batch_size, inner_step, epochs):
    global PARAMETER_INIT, GLOBAL_MAX
    print('-----MAML 训练阶段开始-----')

    # -----创建两类checkpoint-----
    global_optimal_ckpt_path = "checkpoints/train/global_optimal"
    maml_track_ckpt_path = "checkpoints/train/maml_track"

    global_optimal_ckpt = tf.train.Checkpoint(transformer=transformer)
    maml_track_ckpt = tf.train.Checkpoint(transformer=transformer)

    global_optimal_ckpt_manager = tf.train.CheckpointManager(global_optimal_ckpt, global_optimal_ckpt_path,
                                                             max_to_keep=1)
    maml_track_ckpt_manager = tf.train.CheckpointManager(maml_track_ckpt, maml_track_ckpt_path, max_to_keep=1)

    # 提取zero-shot参数
    maml_track_ckpt.restore(maml_track_ckpt_manager.latest_checkpoint)
    if maml_track_ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(maml_track_ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch...")

    # -----Training-----
    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch, batch_content in enumerate(dataset):
            # -----从dataset中取batch数据-----
            end1 = support_set_batch_size
            start2 = end1
            end2 = start2 + query_set_batch_size

            support_set_review_ids = batch_content['review_ids'][0:end1, :]
            query_set_review_ids = batch_content['review_ids'][start2:end2, :]

            support_set_embed_enc_input = batch_content['embed_enc_input'][0:end1, :, :]
            query_set_embed_enc_input = batch_content['embed_enc_input'][start2:end2, :, :]

            support_set_extended_review_ids = batch_content['extended_review_ids'][0:end1, :]
            query_set_extended_review_ids = batch_content['extended_review_ids'][start2:end2, :]

            support_set_batch_review_oov_len = tf.shape(batch_content['review_oov_list'][0:end1, :])[1]
            query_set_batch_review_oov_len = tf.shape(batch_content['review_oov_list'][start2:end2, :])[1]

            support_set_summary_ids = batch_content['summary_ids'][0:end1, :]
            query_set_summary_ids = batch_content['summary_ids'][start2:end2, :]

            support_set_embed_dec_input = batch_content['embed_dec_input'][0:end1, :, :]
            query_set_embed_dec_input = batch_content['embed_dec_input'][start2:end2, :, :]

            support_set_extended_summary_ids = batch_content['extended_summary_ids'][0:end1, :]
            query_set_extended_summary_ids = batch_content['extended_summary_ids'][start2:end2, :]

            support_set_annotation_ids = batch_content['annotation_ids'][0:end1, :]
            query_set_annotation_ids = batch_content['annotation_ids'][start2:end2, :]

            support_set_embed_ann_input = batch_content['embed_ann_input'][0:end1, :, :]
            query_set_embed_ann_input = batch_content['embed_ann_input'][start2:end2, :, :]

            support_set_extended_annotation_ids = batch_content['extended_annotation_ids'][0:end1, :]
            query_set_extended_annotation_ids = batch_content['extended_review_ids'][start2:end2, :]

            support_set_batch_annotation_oov_len = tf.shape(batch_content['annotation_oov_list'][0:end1, :])[1]
            query_set_batch_annotation_oov_len = tf.shape(batch_content['annotation_oov_list'][start2:end2, :])[1]

            support_set_embed_r_y_input = batch_content['embed_r_y_input'][0:end1, :, :]
            query_set_embed_r_y_input = batch_content['embed_r_y_input'][start2:end2, :, :]

            # 无用：仅用于执行一次恢复参数
            if PARAMETER_INIT is not True:
                print("---- 参数恢复 ----")
                transformer(support_set_embed_enc_input, support_set_embed_dec_input, support_set_embed_r_y_input,
                            support_set_embed_ann_input,
                            extended_inp=support_set_extended_review_ids,
                            extended_ann_inp=support_set_extended_annotation_ids,
                            max_r_oov_len=0,
                            max_a_oov_len=0,
                            training=True,
                            enc_padding_mask=None,
                            look_ahead_mask=None,
                            dec_padding_mask=None,
                            look_ahead_mask_a=None,
                            dec_padding_mask_a=None,
                            batch_size=support_set_batch_size,
                            trainable=False,
                            lambda_xy=0,
                            lambda_ay=0)
                PARAMETER_INIT = True

            train_step_maml(transformer, maml_inner_optimizer, maml_outer_optimizer,
                            train_loss, train_accuracy,
                            support_set_review_ids, support_set_embed_enc_input,
                            support_set_extended_review_ids,
                            support_set_batch_review_oov_len,
                            support_set_summary_ids, support_set_embed_dec_input,
                            support_set_extended_summary_ids,
                            support_set_annotation_ids, support_set_embed_ann_input,
                            support_set_extended_annotation_ids,
                            support_set_batch_annotation_oov_len,
                            support_set_embed_r_y_input,
                            query_set_review_ids, query_set_embed_enc_input, query_set_extended_review_ids,
                            query_set_batch_review_oov_len,
                            query_set_summary_ids, query_set_embed_dec_input,
                            query_set_extended_summary_ids,
                            query_set_annotation_ids, query_set_embed_ann_input,
                            query_set_extended_annotation_ids,
                            query_set_batch_annotation_oov_len,
                            query_set_embed_r_y_input,
                            support_set_batch_size,
                            query_set_batch_size,
                            inner_step)

            if (batch + 1) % 50 == 0:
                print(
                    f'Epoch {epoch} Batch {batch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        print(f'Epoch {epoch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        if epoch % 10 == 0:
            maml_track_ckpt_manager.save()
            print("save maml track checkpoint...")
        # maml_track_ckpt_manager.save()

        # -----Fine-tune----- 处理全局最优
        print('---Fine-tune Start...-----')
        for epoch1 in range(10):
            for batch, batch_content in enumerate(dataset2):
                # -----从dataset中取batch数据-----
                review_ids = batch_content['review_ids']  # 传入mask, inp
                summary_ids = batch_content['summary_ids']  # 传入mask, tar_inp
                annotation_ids = batch_content['annotation_ids']  # 传入mask

                embed_enc_input = batch_content['embed_enc_input']  # 传入transformer, encoder的输入
                embed_dec_input = batch_content['embed_dec_input']  # 传入transformer, decoder的输入
                embed_ann_input = batch_content['embed_ann_input']  # 传入transformer, r-net的输入
                embed_r_y_input = batch_content['embed_r_y_input']  # 传入transformer, r-net的输入

                extended_review_ids = batch_content['extended_review_ids']  # 传入transformer, calc_final_dist计算
                extended_summary_ids = batch_content['extended_summary_ids']  # tar_real, loss_function计算
                extended_annotation_ids = batch_content['extended_annotation_ids']  # 传入transformer, calc_final_dist计算

                batch_review_oov_len = tf.shape(batch_content["review_oov_list"])[1]
                batch_annotation_oov_len = tf.shape(batch_content["annotation_oov_list"])[1]

                train_step_fine_tune(transformer, maml_outer_optimizer, train_loss, train_accuracy,
                                     review_ids, embed_enc_input, extended_review_ids, batch_review_oov_len,
                                     summary_ids, embed_dec_input, extended_summary_ids,
                                     annotation_ids, embed_ann_input, extended_annotation_ids,
                                     batch_annotation_oov_len,
                                     embed_r_y_input,
                                     10)

            if (epoch1 + 1) % 10 == 0:
                print(f'Epoch {epoch1 + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        print('---Fine-tune End-----')

        # -----TensorBoard-----
        # tensorboard --logdir logs/gradient_tape
        with train_summary_writer.as_default():
            tf.summary.scalar('MAML Loss', train_loss.result(), step=epoch)
            tf.summary.scalar('MAML Accuracy', train_accuracy.result(), step=epoch)

        # MAML_eval
        _, _, _, ROUGE_AVG, BLEU, METEOR = for_evaluate(transformer, dataset_evaluate, 1)
        current_three_avg = (ROUGE_AVG + BLEU + METEOR) / 3

        if current_three_avg > GLOBAL_MAX:
            print(GLOBAL_MAX, "<-", current_three_avg)
            GLOBAL_MAX = current_three_avg
            global_optimal_ckpt_manager.save()
            print("global optimal epoch: ", epoch)
            print("save global max checkpoint ...")

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')


train_for_maml(dataset_maml, dataset_fine_tune, support_set_batch_size, query_set_batch_size, inner_step=inner_step,
               epochs=TOTAL_EPOCH)
