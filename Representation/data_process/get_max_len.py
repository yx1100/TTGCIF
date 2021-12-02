import json
from transformers import BertTokenizer

'''
统计一下8个域每个域100条评论，最长的一条，token数是多少，即为每个域的max_len
shape: [
  {
    "Arts_Crafts_and_Sewing_review_max_len": 122,
    "CDs_and_Vinyl_review_max_len": 172,
    "Digital_Music_review_max_len": 124,
    "Electronics_review_max_len": 177,
    "Kindle_Store_review_max_len": 125,
    "Movies_and_TV_review_max_len": 123,
    "Toys_and_Games_review_max_len": 109,
    "Video_Games_review_max_len": 141
  },
  {
    "Arts_Crafts_and_Sewing_summary_max_len": 33,
    "CDs_and_Vinyl_summary_max_len": 40,
    "Digital_Music_summary_max_len": 36,
    "Electronics_summary_max_len": 47,
    "Kindle_Store_summary_max_len": 34,
    "Movies_and_TV_summary_max_len": 28,
    "Toys_and_Games_summary_max_len": 0,
    "Video_Games_summary_max_len": 36
  }
]
读取：/数据集/processed/2
保存为max_len_result.json
'''

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


def get_sentient(path):
    review_sent = []
    summary_sent = []
    # for file in files:  # file是一条评论或摘要的pickle文件, files type: list
    print('当前处理文件名：', path + '\n')
    with open(path, 'r') as ff:
        one = json.load(ff)  # one是 一条评论或摘要 分句后的列表，type:list

    for o in one:
        review_sent.append(o['review'])
        summary_sent.append(o['summary'])

    return review_sent, summary_sent


result = []
review_len = {}
summary_len = {}
tokenizer = BertTokenizer.from_pretrained("../bert_uncased_L-8_H-512_A-8")

# 1 遍历各个域
for domain in domains:
    path1 = '../../datasets/4_dataset_a_pseudo/' + domain + '_a_p.json'  # 要 读取 的 domain文件目录
    print('-----' + domain + '-----')

    # 2 将所有的句子（最细分）读出来并加入到sent_list中
    all_review_sent, all_summary_sent = get_sentient(path1)

    print('该domain的评论条数: ', len(all_review_sent))  # 应该是每个域的评论条数:100
    print('该domain的摘要条数: ', len(all_summary_sent))  # 同上

    align_review_encoder_input = tokenizer(all_review_sent, return_tensors='tf', padding=True)
    align_summary_encoder_input = tokenizer(all_summary_sent, return_tensors='tf', padding=True)

    print(align_review_encoder_input.token_type_ids.shape[1])
    print(align_summary_encoder_input.token_type_ids.shape[1])

    review_len[domain + '_review_max_len'] = align_review_encoder_input.token_type_ids.shape[1]
    summary_len[domain + '_summary_max_len'] = align_summary_encoder_input.token_type_ids.shape[1]

result.append(review_len)
result.append(summary_len)
# with open('../max_len_result.json', 'w') as f:
#     json.dump(result, f)
