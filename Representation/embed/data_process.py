from tokenization import convert_tokens_to_ids


def get_review_summary_tokens_ids(review, summary, annotation, pseudo, bert_tokenizer, self_tokenizer, en_max_len,
                                  de_max_len):
    """
    以一句评论和摘要为单位
    :param review: 一条评论文本，type: string
    :param summary: 一条摘要文本，type: string
    :param annotation: 一条摘要标注，type: string
    :param pseudo: 一条伪摘要文本，type: string
    :param bert_tokenizer: 调用BertTokenizer做含unk的tokenization
    :param self_tokenizer: 调用FullTokenizer做含oov的tokenization
    :param en_max_len:
    :param de_max_len:
    :return:
            review_ids:               ids,    unk, 含[CLS]和[SEP]
            extended_review_tokens:   tokens, oov, 含[CLS]和[SEP]
            extended_review_ids:      ids,    oov, 含[CLS]和[SEP]
            summary_ids:              ids,    unk, 含[CLS]和[SEP]
            extended_summary_tokens:  tokens, oov, 含[CLS]和[SEP]
            extended_summary_ids:     ids,    oov, 含[CLS]和[SEP]
            review_oov_list:                 tokens
    """
    # 获取词表vocabs和词表长度vocab_size
    vocabs = bert_tokenizer.vocab  # 词表，type: dict
    vocab_size = bert_tokenizer.vocab_size  # 词表长度30522

    # 调用BertTokenizer做含unk的token→id list; review和summary
    review_ids = bert_tokenizer(review, padding='max_length', max_length=en_max_len, return_tensors='tf').input_ids
    summary_ids1 = bert_tokenizer(summary, padding='max_length', max_length=de_max_len, return_tensors='tf').input_ids
    summary_ids2 = bert_tokenizer(pseudo, padding='max_length', max_length=de_max_len, return_tensors='tf').input_ids
    annotation_ids = bert_tokenizer(annotation, padding='max_length', max_length=en_max_len,
                                    return_tensors='tf').input_ids
    pseudo_ids1 = bert_tokenizer(summary, padding='max_length', max_length=en_max_len, return_tensors='tf').input_ids
    pseudo_ids2 = bert_tokenizer(pseudo, padding='max_length', max_length=en_max_len, return_tensors='tf').input_ids

    # 获取含oov的review token list和oov list
    extended_review_tokens, review_oov_list = self_tokenizer.tokenize(review)
    extended_review_tokens = ['[CLS]'] + extended_review_tokens + ['[SEP]']  # 添加start和end token
    review_oov_list = list(set(review_oov_list))  # oov去重

    extended_annotation_tokens, annotation_oov_list = self_tokenizer.tokenize(annotation)
    extended_annotation_tokens = ['[CLS]'] + extended_annotation_tokens + ['[SEP]']
    annotation_oov_list = list(set(annotation_oov_list))

    # 获取含oov的summary token list和oov list
    extended_summary_tokens, _ = self_tokenizer.tokenize(summary)
    extended_summary_tokens = ['[CLS]'] + extended_summary_tokens + ['[SEP]']  # 添加start和end token

    for oov in review_oov_list:
        vocabs[oov] = vocab_size + review_oov_list.index(oov)  # 词表扩充, type: 字典

    extended_review_ids = convert_tokens_to_ids(vocabs, extended_review_tokens)
    extended_summary_ids = convert_tokens_to_ids(vocabs, extended_summary_tokens)
    extended_annotation_ids = convert_tokens_to_ids(vocabs, extended_annotation_tokens)

    return review_ids, extended_review_ids, review_oov_list, summary_ids, extended_summary_ids, annotation_ids, extended_annotation_ids, annotation_oov_list, pseudo_ids
