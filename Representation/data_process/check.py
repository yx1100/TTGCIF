import json

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

for domain in domains:
    read_file_path = "/Users/yuxin/Developer/TPCC/datasets/4_dataset_a_pseudo/" + domain + '_a_p.json'  # 要读取的domain文件
    with open(read_file_path, 'r') as f:
        content = json.load(f)  # content type:list

    print('-----' + domain + '-----')
    for c in content:  # c type: dict
        review1 = c['review']  # 读取每一条评论
        summary1 = c['summary']
        annotation1 = c['annotation']
        pseudo1 = c['pseudo']

        print(str(len(review1)) + '/' + str(len(summary1)) + '/' + str(len(annotation1)) + '/' + str(len(pseudo1)))

        if len(review1) == 0 or len(summary1) == 0 or len(annotation1) == 0 or len(pseudo1) == 0:
            print(review1 + '/' + summary1 + '/' + annotation1 + '/' + pseudo1)

        if len(summary1) > len(review1) or len(annotation1) > len(review1) or len(pseudo1) > len(review1):
            print(review1)

            print(summary1)

            print(annotation1)

            print(pseudo1)
            print('----------')
