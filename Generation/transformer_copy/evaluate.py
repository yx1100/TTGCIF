import sys
import tensorflow as tf

from numpy import mean
from rouge import Rouge
from transformers import BertTokenizer
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from train_step import train_step_evaluate


def get_evaluate(prediction, ground_truth):
    """
    :param prediction: 可以是list/tensor
    :param ground_truth: 同上
    :return:
    """
    tokenizer = BertTokenizer.from_pretrained('./bert_uncased_L-8_H-512_A-8')
    prediction = tokenizer.batch_decode(prediction, skip_special_tokens=True)
    ground_truth = tokenizer.batch_decode(ground_truth, skip_special_tokens=True)

    rouge = Rouge()

    rouge_1_sum = 0
    rouge_2_sum = 0
    rouge_l_sum = 0
    rouge_avg_sum = 0
    bleu_score_sum = 0
    metero_score_sum = 0
    nums = 0
    pred_list = []
    ground_list = []
    for pred, ground in zip(prediction, ground_truth):
        nums = nums + 1
        pred_list.append(pred)
        print(pred_list)
        ground_list.append(ground)

        try:
            rouge_score = rouge.get_scores(pred, ground)
            rouge_1 = rouge_score[0]["rouge-1"]['f']
            rouge_1_sum = rouge_1_sum + rouge_1
            rouge_2 = rouge_score[0]["rouge-2"]['f']
            rouge_2_sum = rouge_2_sum + rouge_2
            rouge_l = rouge_score[0]["rouge-l"]['f']
            rouge_l_sum = rouge_l_sum + rouge_l
            rouge_avg = (rouge_1 + rouge_2 + rouge_l) / 3
            rouge_avg_sum = rouge_avg_sum + rouge_avg
        except ValueError:
            rouge_1 = 0
            rouge_1_sum = rouge_1_sum + rouge_1
            rouge_2 = 0
            rouge_2_sum = rouge_2_sum + rouge_2
            rouge_l = 0
            rouge_l_sum = rouge_l_sum + rouge_l
            rouge_avg = 0
            rouge_avg_sum = rouge_avg_sum + rouge_avg

        pred_tokenize = word_tokenize(pred)
        ground_tokenize = word_tokenize(ground)
        bleu_score = sentence_bleu([pred_tokenize], ground_tokenize)  # weights决定n-grams
        bleu_score_sum = bleu_score_sum + bleu_score

        metero_score = meteor_score([ground], pred)

        metero_score_sum = metero_score_sum + metero_score

    # ------ROUGE-----
    rouge_1 = rouge_1_sum / nums
    rouge_2 = rouge_2_sum / nums
    rouge_l = rouge_l_sum / nums
    rouge_avg = rouge_avg_sum / nums
    # print('rouge_1: ', rouge_1)
    # print('rouge_2: ', rouge_2)
    # print('rouge_l: ', rouge_l)
    # print('rouge_avg: ', rouge_avg)
    # ------BLEU-----
    bleu_score = bleu_score_sum / nums
    # print('bleu_score: ', bleu_score)
    # ------METERO-----
    metero_score = metero_score_sum / nums
    # print('metero_score: ', metero_score)

    save_result = {
        'pr': pred_list,
        'gr': ground_list,
        'r1': rouge_1,
        'r2': rouge_2,
        'rl': rouge_l,
        'ra': rouge_avg,
        'bl': bleu_score,
        'me': metero_score
    }

    # try:
    #     f = open('./score.json')
    #     f.close()
    # except FileNotFoundError:
    #     with open('./score.json', 'w') as file:
    #         json.dump([save_result], file)
    #     file.close()
    # else:
    #     with open('./score.json', 'r') as file:
    #         j = json.load(file)
    #     j.append(save_result)
    #     file.close()
    #
    #     with open('./score.json', 'w') as file:
    #         json.dump(j, file)
    #     file.close()

    return rouge_1, rouge_2, rouge_l, rouge_avg, bleu_score, metero_score


def for_evaluate(transformer, dataset, batch_size):
    print('---Starting Evaluate-----')
    # # -----CheckPoint-----
    # checkpoint_path = "checkpoints/train"
    # ckpt = tf.train.Checkpoint(transformer=transformer)
    # ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
    #
    # # if a checkpoint exists, restore the latest checkpoint.
    # ckpt.restore(ckpt_manager.latest_checkpoint)
    # if ckpt_manager.latest_checkpoint:
    #     print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    # else:
    #     print("Error...")
    #     sys.exit(0)

    r1 = []
    r2 = []
    rl = []
    ra = []
    bl = []
    me = []
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

        pred, real = train_step_evaluate(transformer,
                                         review_ids, embed_enc_input, extended_review_ids, batch_review_oov_len,
                                         summary_ids, embed_dec_input, extended_summary_ids,
                                         annotation_ids, embed_ann_input, extended_annotation_ids,
                                         batch_annotation_oov_len,
                                         embed_r_y_input,
                                         batch_size)

        # if (epoch + 1) % 10 == 0:
        predicted_ids = tf.argmax(pred, axis=-1, output_type=tf.int32)
        rouge_1, rouge_2, rouge_l, rouge_avg, bleu_score, metero_score = get_evaluate(predicted_ids, real)
        r1.append(rouge_1)
        r2.append(rouge_2)
        rl.append(rouge_l)
        ra.append(rouge_avg)
        bl.append(bleu_score)
        me.append(metero_score)

    mean_r1 = mean(r1)
    mean_r2 = mean(r2)
    mean_rl = mean(rl)
    mean_ra = mean(ra)
    mean_bl = mean(bl)
    mean_me = mean(me)

    print('r1: ', mean_r1)
    print('r2: ', mean_r2)
    print('rl: ', mean_rl)
    print('ra: ', mean_ra)
    print('bl: ', mean_bl)
    print('me: ', mean_me)

    print('---Evaluate Success---\n')

    return mean_r1, mean_r2, mean_rl, mean_ra, mean_bl, mean_me
