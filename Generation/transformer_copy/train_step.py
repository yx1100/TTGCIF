import tensorflow as tf
from Generation.transformer_copy.module.utils import create_masks
from Generation.transformer_copy.module.loss_and_metrics import loss_function, accuracy_function


@tf.function
def train_step_zeroshot(transformer, optimizer, train_loss, train_accuracy,
                        review_ids, embed_enc_input, extended_review_ids, batch_review_oov_len,
                        summary_ids, embed_dec_input, extended_summary_ids,
                        annotation_ids, embed_ann_input, extended_annotation_ids, batch_annotation_oov_len,
                        embed_r_y_input,
                        batch_size,
                        total_lambda=0):
    # 目标（target）被分成了 tar_inp 和 tar_real。
    # tar_inp 作为输入传递到解码器。
    # tar_real 是位移了 1 的同一个输入：在 tar_inp 中的每个位置，tar_real 包含了应该被预测到的下一个标记（token）。
    # sentence = "SOS A lion in the jungle is sleeping EOS"
    # tar_inp = "SOS A lion in the jungle is sleeping"
    # tar_real = "A lion in the jungle is sleeping EOS"
    inp = review_ids
    tar_inp = summary_ids
    tar_real = extended_summary_ids

    # ---review mask---
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    # inp和tar_inp是embedding后的表示
    x_inp = embed_enc_input
    y_tar = embed_dec_input
    y_inp = embed_r_y_input
    a_inp = embed_ann_input

    extended_review_ids = extended_review_ids
    batch_review_oov_len = batch_review_oov_len
    extended_annotation_ids = extended_annotation_ids
    batch_annotation_oov_len = batch_annotation_oov_len
    # ---annotation mask---
    _, combined_mask_a, dec_padding_mask_a = create_masks(annotation_ids, summary_ids)

    with tf.GradientTape() as tape:
        # predictions shape: (batch_size, tar_seq_len, vocab_size)
        predictions, r_a_output, dis1, dis2 = transformer(x_inp, y_tar, y_inp, a_inp,
                                                          extended_inp=extended_review_ids,
                                                          extended_ann_inp=extended_annotation_ids,
                                                          max_r_oov_len=batch_review_oov_len,
                                                          max_a_oov_len=batch_annotation_oov_len,
                                                          training=True,
                                                          enc_padding_mask=enc_padding_mask,
                                                          look_ahead_mask=combined_mask,
                                                          dec_padding_mask=dec_padding_mask,
                                                          look_ahead_mask_a=combined_mask_a,
                                                          dec_padding_mask_a=dec_padding_mask_a,
                                                          batch_size=batch_size,
                                                          lambda_xy=1,
                                                          lambda_ay=1)

        loss_xy = loss_function(tar_real, predictions) + dis1
        loss_ay = loss_function(tar_real, r_a_output) + dis2

        if total_lambda == 0:
            total_loss = loss_xy
        elif total_lambda == 1:
            total_loss = loss_xy + total_lambda * loss_ay

    gradients = tape.gradient(total_loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(total_loss)
    train_accuracy(accuracy_function(tar_real, predictions))

    return predictions, tar_real


# @tf.function
def train_step_maml(transformer, inner_optimizer, outer_optimizer, train_loss, train_accuracy,
                    support_set_review_ids, support_set_embed_enc_input, support_set_extended_review_ids,
                    support_set_batch_review_oov_len,
                    support_set_summary_ids, support_set_embed_dec_input, support_set_extended_summary_ids,
                    support_set_annotation_ids, support_set_embed_ann_input, support_set_extended_annotation_ids,
                    support_set_batch_annotation_oov_len,
                    support_set_embed_r_y_input,
                    query_set_review_ids, query_set_embed_enc_input, query_set_extended_review_ids,
                    query_set_batch_review_oov_len,
                    query_set_summary_ids, query_set_embed_dec_input, query_set_extended_summary_ids,
                    query_set_annotation_ids, query_set_embed_ann_input, query_set_extended_annotation_ids,
                    query_set_batch_annotation_oov_len,
                    query_set_embed_r_y_input,
                    support_set_batch_size,
                    query_set_batch_size,
                    inner_step, trainable=False):
    support_set_inp = support_set_review_ids
    support_set_tar_inp = support_set_summary_ids
    support_set_tar_real = support_set_extended_summary_ids

    # ---review mask---
    support_set_enc_padding_mask, support_set_combined_mask, support_set_dec_padding_mask = create_masks(
        support_set_inp, support_set_tar_inp)

    # inp和tar_inp是embedding后的表示
    support_set_x_inp = support_set_embed_enc_input
    support_set_y_tar = support_set_embed_dec_input
    support_set_y_inp = support_set_embed_r_y_input
    support_set_a_inp = support_set_embed_ann_input

    support_set_extended_review_ids = support_set_extended_review_ids
    support_set_batch_review_oov_len = support_set_batch_review_oov_len
    support_set_extended_annotation_ids = support_set_extended_annotation_ids
    support_set_batch_annotation_oov_len = support_set_batch_annotation_oov_len
    # ---annotation mask---
    _, support_set_combined_mask_a, support_set_dec_padding_mask_a = create_masks(support_set_annotation_ids,
                                                                                  support_set_summary_ids)

    # --------------------
    query_set_inp = query_set_review_ids
    query_set_tar_inp = query_set_summary_ids
    query_set_tar_real = query_set_extended_summary_ids

    # ---review mask---
    query_set_enc_padding_mask, query_set_combined_mask, query_set_dec_padding_mask = create_masks(query_set_inp,
                                                                                                   query_set_tar_inp)

    # inp和tar_inp是embedding后的表示
    query_set_x_inp = query_set_embed_enc_input
    query_set_y_tar = query_set_embed_dec_input
    query_set_y_inp = query_set_embed_r_y_input
    query_set_a_inp = query_set_embed_ann_input

    query_set_extended_review_ids = query_set_extended_review_ids
    query_set_batch_review_oov_len = query_set_batch_review_oov_len
    query_set_extended_annotation_ids = query_set_extended_annotation_ids
    query_set_batch_annotation_oov_len = query_set_batch_annotation_oov_len
    # ---annotation mask---
    _, query_set_combined_mask_a, query_set_dec_padding_mask_a = create_masks(query_set_annotation_ids,
                                                                              query_set_summary_ids)

    model_weight = transformer.get_weights()

    for _ in range(inner_step):
        with tf.GradientTape() as tape:
            predictions, _, _, _ = transformer(support_set_x_inp, support_set_y_tar, support_set_y_inp,
                                               support_set_a_inp,
                                               extended_inp=support_set_extended_review_ids,
                                               extended_ann_inp=support_set_extended_annotation_ids,
                                               max_r_oov_len=support_set_batch_review_oov_len,
                                               max_a_oov_len=support_set_batch_annotation_oov_len,
                                               training=True,
                                               enc_padding_mask=support_set_enc_padding_mask,
                                               look_ahead_mask=support_set_combined_mask,
                                               dec_padding_mask=support_set_dec_padding_mask,
                                               look_ahead_mask_a=support_set_combined_mask_a,
                                               dec_padding_mask_a=support_set_dec_padding_mask_a,
                                               batch_size=support_set_batch_size,
                                               trainable=trainable)

            loss = loss_function(support_set_tar_real, predictions)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        inner_optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    with tf.GradientTape() as tape:
        predictions, _, _, _ = transformer(query_set_x_inp, query_set_y_tar, query_set_y_inp, query_set_a_inp,
                                           extended_inp=query_set_extended_review_ids,
                                           extended_ann_inp=query_set_extended_annotation_ids,
                                           max_r_oov_len=query_set_batch_review_oov_len,
                                           max_a_oov_len=query_set_batch_annotation_oov_len,
                                           training=True,
                                           enc_padding_mask=query_set_enc_padding_mask,
                                           look_ahead_mask=query_set_combined_mask,
                                           dec_padding_mask=query_set_dec_padding_mask,
                                           look_ahead_mask_a=query_set_combined_mask_a,
                                           dec_padding_mask_a=query_set_dec_padding_mask_a,
                                           batch_size=query_set_batch_size,
                                           trainable=trainable)

        loss = loss_function(query_set_tar_real, predictions)
        loss = tf.reduce_mean(loss)

    transformer.set_weights(model_weight)
    grads = tape.gradient(loss, transformer.trainable_variables)
    outer_optimizer.apply_gradients(zip(grads, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(query_set_tar_real, predictions))

    return predictions, query_set_tar_real


@tf.function
def train_step_fine_tune(transformer, outer_optimizer, train_loss, train_accuracy,
                         query_set_review_ids, query_set_embed_enc_input, query_set_extended_review_ids,
                         query_set_batch_review_oov_len,
                         query_set_summary_ids, query_set_embed_dec_input, query_set_extended_summary_ids,
                         query_set_annotation_ids, query_set_embed_ann_input, query_set_extended_annotation_ids,
                         query_set_batch_annotation_oov_len,
                         query_set_embed_r_y_input,
                         query_set_batch_size,
                         trainable=False):
    # --------------------
    query_set_inp = query_set_review_ids
    query_set_tar_inp = query_set_summary_ids
    query_set_tar_real = query_set_extended_summary_ids

    # ---review mask---
    query_set_enc_padding_mask, query_set_combined_mask, query_set_dec_padding_mask = create_masks(query_set_inp,
                                                                                                   query_set_tar_inp)

    # inp和tar_inp是embedding后的表示
    query_set_x_inp = query_set_embed_enc_input
    query_set_y_tar = query_set_embed_dec_input
    query_set_y_inp = query_set_embed_r_y_input
    query_set_a_inp = query_set_embed_ann_input

    query_set_extended_review_ids = query_set_extended_review_ids
    query_set_batch_review_oov_len = query_set_batch_review_oov_len
    query_set_extended_annotation_ids = query_set_extended_annotation_ids
    query_set_batch_annotation_oov_len = query_set_batch_annotation_oov_len
    # ---annotation mask---
    _, query_set_combined_mask_a, query_set_dec_padding_mask_a = create_masks(query_set_annotation_ids,
                                                                              query_set_summary_ids)

    with tf.GradientTape() as tape:
        predictions, _, _, _ = transformer(query_set_x_inp, query_set_y_tar, query_set_y_inp, query_set_a_inp,
                                           extended_inp=query_set_extended_review_ids,
                                           extended_ann_inp=query_set_extended_annotation_ids,
                                           max_r_oov_len=query_set_batch_review_oov_len,
                                           max_a_oov_len=query_set_batch_annotation_oov_len,
                                           training=True,
                                           enc_padding_mask=query_set_enc_padding_mask,
                                           look_ahead_mask=query_set_combined_mask,
                                           dec_padding_mask=query_set_dec_padding_mask,
                                           look_ahead_mask_a=query_set_combined_mask_a,
                                           dec_padding_mask_a=query_set_dec_padding_mask_a,
                                           batch_size=query_set_batch_size,
                                           trainable=trainable)

        loss = loss_function(query_set_tar_real, predictions)

    grads = tape.gradient(loss, transformer.trainable_variables)
    outer_optimizer.apply_gradients(zip(grads, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(query_set_tar_real, predictions))

    return predictions, query_set_tar_real


def train_step_evaluate(transformer,
                        review_ids, embed_enc_input, extended_review_ids,
                        batch_review_oov_len,
                        summary_ids, embed_dec_input, extended_summary_ids,
                        annotation_ids, embed_ann_input, extended_annotation_ids,
                        batch_annotation_oov_len,
                        embed_r_y_input,
                        batch_size,
                        trainable=False):
    # --------------------
    query_set_inp = review_ids
    query_set_tar_inp = summary_ids
    tar_real = extended_summary_ids

    # ---review mask---
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(query_set_inp, query_set_tar_inp)

    # inp和tar_inp是embedding后的表示
    query_set_x_inp = embed_enc_input
    query_set_y_tar = embed_dec_input
    query_set_y_inp = embed_r_y_input
    query_set_a_inp = embed_ann_input

    extended_review_ids = extended_review_ids
    batch_review_oov_len = batch_review_oov_len
    extended_annotation_ids = extended_annotation_ids
    batch_annotation_oov_len = batch_annotation_oov_len
    # ---annotation mask---
    _, combined_mask_a, dec_padding_mask_a = create_masks(annotation_ids, summary_ids)

    predictions = transformer(query_set_x_inp, query_set_y_tar, query_set_y_inp, query_set_a_inp,
                              extended_inp=extended_review_ids,
                              extended_ann_inp=extended_annotation_ids,
                              max_r_oov_len=batch_review_oov_len,
                              max_a_oov_len=batch_annotation_oov_len,
                              training=False,
                              enc_padding_mask=enc_padding_mask,
                              look_ahead_mask=combined_mask,
                              dec_padding_mask=dec_padding_mask,
                              look_ahead_mask_a=combined_mask_a,
                              dec_padding_mask_a=dec_padding_mask_a,
                              batch_size=batch_size,
                              trainable=trainable)

    return predictions, tar_real
