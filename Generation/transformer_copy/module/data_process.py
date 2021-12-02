from Generation.transformer_copy.module.tokenization import convert_tokens_to_ids


def get_review_summary_tokens_ids(review, summary, bert_tokenizer, self_tokenizer, enc_max_len, dec_max_len):
    """
    以一句评论和摘要为单位
    :param review: 一条评论文本，type: string
    :param summary: 一条摘要文本，type: string
    :param bert_tokenizer: 调用BertTokenizer做含unk的tokenization
    :param self_tokenizer: 调用FullTokenizer做含oov的tokenization
    :return:
            review_tokens:            tokens, unk, 不含[CLS]和[SEP]
            review_ids:               ids,    unk, 含[CLS]和[SEP]
            extended_review_tokens:   tokens, oov, 含[CLS]和[SEP]
            extended_review_ids:      ids,    oov, 含[CLS]和[SEP]
            summary_tokens:           tokens, unk, 不含[CLS]和[SEP]
            summary_ids:              ids,    unk, 含[CLS]和[SEP]
            extended_summary_tokens:  tokens, oov, 含[CLS]和[SEP]
            extended_summary_ids:     ids,    oov, 含[CLS]和[SEP]
            oov_list:                 tokens
    """
    # 获取词表vocabs和词表长度vocab_size
    vocabs = bert_tokenizer.vocab  # 词表，type: dict
    vocab_size = bert_tokenizer.vocab_size  # 词表长度30522

    # 调用BertTokenizer做含unk的token→id list; review和summary
    review_tokens = bert_tokenizer.tokenize(review)
    review_ids = bert_tokenizer(review, padding='max_length', max_length=enc_max_len).input_ids  # star

    summary_tokens = bert_tokenizer.tokenize(summary)
    summary_ids = bert_tokenizer(summary, padding='max_length', max_length=dec_max_len).input_ids  # star

    # 获取含oov的review token list和oov list
    extended_review_tokens, oov_list = self_tokenizer.tokenize(review)
    extended_review_tokens = ['[CLS]'] + extended_review_tokens + ['[SEP]']  # 添加start和end token
    oov_list = list(set(oov_list))  # oov去重 star

    # 获取含oov的summary token list和oov list
    extended_summary_tokens, _ = self_tokenizer.tokenize(summary)
    extended_summary_tokens = ['[CLS]'] + extended_summary_tokens + ['[SEP]']  # 添加start和end token

    for oov in oov_list:
        vocabs[oov] = vocab_size + oov_list.index(oov)  # 词表扩充, type: 字典

    extended_review_ids = convert_tokens_to_ids(vocabs, extended_review_tokens)  # star
    extended_summary_ids = convert_tokens_to_ids(vocabs, extended_summary_tokens)  # star

    # return review_tokens, review_ids, extended_review_tokens, extended_review_ids, summary_tokens, summary_ids, extended_summary_tokens, extended_summary_ids, oov_list
    return review_ids, extended_review_ids, summary_ids, extended_summary_ids, oov_list
