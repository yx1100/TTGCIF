import sys
import time
import tensorflow as tf

from datetime import datetime
from transformers import BertTokenizer

from model.transformer import Transformer
from module.optimizer import CustomSchedule
from module.loss_and_metrics import train_loss, train_accuracy
from train_step import train_step_zeroshot
from input_pipeline_zeroshot import dataset_zeroshot, BATCH_SIZE_OF_ZEROSHOT
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

print('-----开始训练-----')

# -----Optimizer-----
lr_shrink_factor = 10
learning_rate = CustomSchedule(d_model, shrink_factor=lr_shrink_factor)
zeroshot_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

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


def train_for_zeroshot(dataset, batch_size, total_lambda, epochs=1000):
    print('-----Zero-shot Learning 训练阶段开始-----')
    # -----CheckPoint-----
    checkpoint_path = "checkpoints/train"
    ckpt = tf.train.Checkpoint(transformer=transformer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

    # if a checkpoint exists, restore the latest checkpoint.
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch....")

    # -----Training-----
    for epoch in range(epochs):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch, batch_content in enumerate(dataset):
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

            # pred是训练时生产的id list, real是对应的ground_truth
            train_step_zeroshot(transformer, zeroshot_optimizer, train_loss, train_accuracy,
                                review_ids, embed_enc_input, extended_review_ids, batch_review_oov_len,
                                summary_ids, embed_dec_input, extended_summary_ids,
                                annotation_ids, embed_ann_input, extended_annotation_ids,
                                batch_annotation_oov_len,
                                embed_r_y_input,
                                batch_size,
                                total_lambda)

            if (batch + 1) % 25 == 0:
                print(
                    f'Epoch {epoch + 1} Batch {batch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        # -----TensorBoard-----
        # tensorboard --logdir ./Generation/transformer_copy/logs/gradient_tape
        with train_summary_writer.as_default():
            tf.summary.scalar('Zero-Shot Learning Loss', train_loss.result(), step=epoch)
            tf.summary.scalar('Zero-Shot Learning Accuracy', train_accuracy.result(), step=epoch)

        if (epoch + 1) % 10 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint at {ckpt_save_path}')

        # 整个target
        if (epoch + 1) % 10 == 0:
            for_evaluate(transformer, dataset_evaluate, 1)
        # for_evaluate(transformer, dataset_evaluate, 1)

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')



train_for_zeroshot(dataset_zeroshot, BATCH_SIZE_OF_ZEROSHOT, total_lambda=1, epochs=1000)
for_evaluate(transformer, dataset_evaluate, 1)
