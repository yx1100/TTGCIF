import json
import nltk
from nltk.corpus import stopwords

"""
tag:
名词：NN/NNS/NNP/NNPS
动词：VB/VBD/VBG/VBN/VBP/VBZ
形容词：JJ/JJR/JJS
副词：RB/RBR/RBS
读取：/数据集/处理后的数据集/1/        每条评论文本
保存：/数据集/处理后的数据集/2/a_inp/  每条评论对应的标注
"""

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


def get_pos_tag(text):
    # 分词
    text = text.lower()
    text_list = nltk.word_tokenize(text)
    # 去掉标点符号
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    text_list = [word for word in text_list if word not in english_punctuations]
    # 去掉停用词
    stops = set(stopwords.words("english"))
    text_list = [word for word in text_list if word not in stops]

    n = nltk.pos_tag(text_list)

    return n


dic1 = {}

for domain in domains:
    reviews = []
    path1 = '../../datasets/2_dataset_5b/' + domain + '_5b.json'  # 要读取的domain文件
    path2 = '../../datasets/3_dataset_annotation/' + domain + '_a.json'  # 要保存的domain文件
    with open(path1, 'r') as f:
        content = json.load(f)

    sents = []
    for c in content:  # 遍历每个域的评论/摘要对
        review = c['review']  # 读取每一条评论
        poses = get_pos_tag(review)  # poses 是review做分词后每个词和对应的pos
        seq = []
        for pos in poses:  # pos是每个(word, pos), type: 元组
            tag = pos[1]  # 取pos
            if tag == 'NN' or tag == 'NNS' or tag == 'NNP' or tag == 'NNPS' or tag == 'VB' or tag == 'VBD' or tag == 'VBG' or tag == 'VBN' or tag == 'VBP' or tag == 'VBZ' or tag == 'JJ' or tag == 'JJR' or tag == 'JJS' or tag == 'RB' or tag == 'RBR' or tag == 'RBS':
                seq.append(pos[0])  # 取对应word

        sent = ' '.join(seq)
        dic1 = {
            'review': review,
            'summary': c['summary'],
            'annotation': sent
        }
        sents.append(dic1)

    with open(path2, 'w') as ff:
        json.dump(sents, ff)

    print('-----' + domain + ' end-----')
