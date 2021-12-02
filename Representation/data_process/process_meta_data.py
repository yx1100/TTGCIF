import json
import random

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
    print('---' + domain + '---')

    open_dir = '../../datasets/1_meta_dataset/' + domain + '_5.json'

    save_dir = '../../datasets/2_dataset_5b/' + domain + '_5b.json'

    dataset = []
    with open(open_dir, 'r') as f:
        for line in f:
            content = json.loads(line)  # 每一个content是一个字典，取其中reviewText和summary

            try:
                {'review': content['reviewText'],
                 'summary': content['summary']}
            except KeyError:
                continue
            else:
                dataset.append({'review': content['reviewText'],
                                'summary': content['summary']})

    print(len(dataset))

    dataset.sort(key=lambda i: len(i['summary']), reverse=True)

    ll = []
    for d in dataset:
        if len(ll) == 500:
            break
        if d['summary'].endswith('...'):
            continue
        if 'www.' in d['review'] or 'www.' in d['summary'] or '<a' in d['review'] or '<a' in d['summary'] or '<div' in \
                d['review'] or '<div' in d['summary'] or '=' in d['review'] or '=' in d['summary'] or 'http' in \
                d['review'] or 'http' in d['summary']:
            continue
        if (len(d['review']) - len(d['summary'])) < 200:
            continue
        # if len(d['summary']) < 100:
        #     continue
        if len(d['review']) > 400:
            continue
        if d not in ll:
            ll.append(d)

    random.shuffle(ll)
    print(len(ll))
    with open(save_dir, 'w') as ff:
        json.dump(ll, ff)
