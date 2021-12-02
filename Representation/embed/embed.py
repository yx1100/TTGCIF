import json
import pickle
from tqdm import trange

from transformers import BertTokenizer, TFBertModel
from tokenization import FullTokenizer
from data_process import get_review_summary_tokens_ids
from Representation.数据处理.get_max_len import result

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


# 3
def save_embed(path, save_path, enc_max_len, dec_max_len):
    bert_tokenizer = BertTokenizer.from_pretrained("../bert_uncased_L-8_H-512_A-8", from_pt=True)
    self_tokenizer = FullTokenizer('../bert_uncased_L-8_H-512_A-8/vocab.txt')
    model = TFBertModel.from_pretrained("../bert_uncased_L-8_H-512_A-8", from_pt=True)

    print('当前处理文件名：', path + '\n')
    with open(path, 'r') as ff:
        contents = json.load(ff)  # contents 是 评论/摘要/标注对，type：list[dict{string}]

    # random_num = random.sample(range(0, 4000), 200)
    # c = []
    # for index in random_num:
    #     c.append(contents[index])
    # contents = c

    ddd = []
    for _, c in zip(trange(len(contents)), contents):  # 读取每一个评论/摘要/标注对 c type:dict
        review = c['review']  # review type:string
        summary = c['summary']  # summary type:string
        annotation = c['annotation']  # annotation type: string
        pseudo = c['pseudo']  # pseudo type: string

        review_ids, extended_review_ids, review_oov_list, summary_ids, extended_summary_ids, annotation_ids, extended_annotation_ids, annotation_oov_list, pseudo_ids = get_review_summary_tokens_ids(
            review, summary, annotation, pseudo,
            bert_tokenizer, self_tokenizer,
            enc_max_len, dec_max_len)
        """
        review_ids              shape (1, enc_max_len);     type: Tensor, ids; padding==True
        extended_review_ids     shape (review_seq_len);     type: list, ids;   padding==False
        review_oov_list         shape (oov_len);            type: list, str;   padding==None
        summary_ids             shape (1, dec_max_len);     type: Tensor, ids; padding==True
        extended_summary_ids    shape (summary_seq_len);    type: list, ids;   padding==False
        annotation_ids          shape (1, enc_max_len);     type: Tensor, ids; padding==true
        extended_annotation_ids shape (annotation_seq_len); type: list, ids;   padding==False
        pseudo_ids              shape (1, enc_max_len);     type: Tensor, ids; padding==true
        """

        # x_inp
        review_ids = review_ids
        embed_enc_input = model(review_ids).last_hidden_state
        extended_review_ids = extended_review_ids
        review_oov_list = review_oov_list
        # y_tar
        summary_ids = summary_ids[:, :-1]
        embed_dec_input = model(summary_ids).last_hidden_state
        extended_summary_ids = extended_summary_ids[1:]
        # a_inp
        annotation_ids = annotation_ids
        embed_ann_input = model(annotation_ids).last_hidden_state
        extended_annotation_ids = extended_annotation_ids
        annotation_oov_list = annotation_oov_list
        # y_inp
        embed_r_y_input = model(pseudo_ids).last_hidden_state

        #  现在每个域都有y_tar和y_inp，y_tar是ground truth，y_inp是伪摘要；
        #  这样相当于在训练的时候，源域的要用y_tar，目标域要用y_inp；
        #  改之前y_inp是summary_ids词嵌入化得到的，是所有域都用的y_inp

        """
        x_inp:
            review_ids              shape (1, enc_max_len);        type: Tensor, ids; padding==True;  transformer之前，传入create_masks
            embed_enc_input         shape (1, enc_max_len, dim);   type: Tensor, ids; padding==True;  传入encoder的输入
            extended_review_ids     shape (review_seq_len);        type: list, ids;   padding==False; 传入transformer，计算calc_final_dist
            review_oov_list         shape (oov_len);               type: list, str;   padding==None;  传入transformer, 需要对齐后取对齐后长度
        y_tar:                                                     
            summary_ids             shape (1, dec_max_len-1);      type: Tensor, ids; padding==True;  transformer之前，传入create_masks，已经做过right shift
            embed_dec_input         shape (1, dec_max_len-1, dim); type: Tensor, ids; padding==True;  传入decoder的输入
            extended_summary_ids    shape (summary_seq_len-1);     type: list, ids;   padding==False;
        a_inp:
            annotation_ids          shape (1, enc_max_len);        type: Tensor, ids; padding==true;
            embed_ann_input         shape (1, enc_max_len, dim);   type: Tensor, ids; padding==True;  传入encoder的输入
            extended_annotation_ids shape (annotation_seq_len);    type: list, ids;   padding==False; 
            annotation_oov_list     shape (oov_len);               type: list, str;   padding==None;  传入transformer, 需要对齐后取对齐后长度
        y_inp:
            embed_r_y_input         shape (1, enc_max_len, dim);   type: Tensor, ids; padding==true;
        """
        big_dict = {
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

        ddd.append(big_dict)

    with open(save_path, 'wb') as fff:
        pickle.dump(ddd, fff)


# 1 获取max_len的值
# with open('../max_len_result.json', 'r') as f:
#     content = json.load(f)
content = result
encoder_max_len = max(content[0].values())
decoder_max_len = max(content[1].values())
print('encoder_max_len: ', encoder_max_len)
print('decoder_max_len: ', decoder_max_len)
# encoder_max_len:  138
# decoder_max_len:  77

# 2 遍历域
for domain in domains:
    read_file_dir = '../../datasets/4_dataset_a_pseudo/' + domain + '_a_p.json'  # 读取数据 的domain文件夹目录
    save_file_dir = '../../datasets/5_dataset_final/' + domain + '.pickle'  # 存放数据 的文件夹目录

    save_embed(read_file_dir, save_file_dir, encoder_max_len, decoder_max_len)

print('-----🚩 处理结束-----')
