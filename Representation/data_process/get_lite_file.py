import json
import random

"""
每个域，从8千条数据中随机抽取100条评论和对应的摘要
type: list
shape: [
        {'reviewText': '...', 
        'summary': '...'}, 
        ...]
读取：数据集/Amazon_reviews/
保存到：数据集/Amazon_reviews_lite/
"""

filenames = ['Arts_Crafts_and_Sewing',
             'CDs_and_Vinyl',
             'Digital_Music',
             'Electronics',
             'Kindle_Store',
             'Movies_and_TV',
             'Toys_and_Games',
             'Video_Games']

for name in filenames:
    path1 = '../数据集/Amazon_reviews/' + name + '.json'
    print(path1)

    with open(path1, 'r') as f:
        content = json.load(f)

    print(len(content))
    L1 = random.sample(range(0, 5900), 100)
    print(L1)

    reviews = []
    for i in L1:
        reviews.append(content[i])

    path2 = '../数据集/Amazon_reviews_lite/' + name + '.json'
    with open(path2, 'w') as ff:
        json.dump(reviews, ff)
