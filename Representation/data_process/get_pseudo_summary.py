import json
import re

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


# 计算在源域上的压缩率
def get_compressibility():
    compress = 0
    i = 0

    for domain in domains:
        domain_file = '../../datasets/2_dataset_5b/' + domain + '_5b.json'  # 要读取的domain文件
        with open(domain_file, 'r') as file:
            content1 = json.load(file)

        for c1 in content1:
            review = c1['review']
            summary = c1['summary']

            compress = compress + (len(summary) / len(review))
            i = i + 1

    return compress / i


compressibility = get_compressibility()
print('压缩率: {}\n'.format(compressibility))

for domain in domains:
    read_file_path = '../../datasets/3_dataset_annotation/' + domain + '_a.json'  # 要读取的domain文件
    save_file_path = '../../datasets/4_dataset_a_pseudo/' + domain + '_a_p.json'  # 要保存的domain文件
    with open(read_file_path, 'r') as f:
        content = json.load(f)  # content type:list

    pseudo_summary = []

    for c in content:  # c type: dict
        review1 = c['review']  # 读取每一条评论
        summary1 = c['summary']
        annotation1 = c['annotation']
        print('review: {}'.format(review1))

        sentences = re.split(r'([,.!?])', review1)

        s = ''
        for sent in sentences:
            if len(s + sent) / len(review1) > compressibility:
                if s == '':
                    s = sent
                else:
                    break
            else:
                if s == '':
                    s = sent
                else:
                    s = s + sent

        print('伪摘要: {}\n'.format(s))
        pseudo_summary.append({
            'review': review1,
            'summary': summary1,
            'annotation': annotation1,
            'pseudo': s
        })

    with open(save_file_path, 'w') as ff:
        json.dump(pseudo_summary, ff)

    print('-----' + domain + ' end-----')
