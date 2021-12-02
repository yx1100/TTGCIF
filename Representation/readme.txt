1. 将 meta_data 文件夹中的原始数据集通过 process_meta.py 处理，生成 dataset_8k 文件夹中每个域8000条数据的数据集
2. 将 dataset_8k 文件夹中每个域的8k条评论通过 get_annotation.py 添加标注，生成 dataset_annotation 中的数据集
3. 将 dataset_annotation 文件夹中的数据集，通过 get_pseudo_summary.py 添加伪摘要，生成 dataset_a_pseudo 文件夹中的数据集
4. 读取 dataset_a_pseudo 文件夹中的数据集，通过 get_max_len.py 统计出每个域中的评论和摘要的 token 数
5. 读取 dataset_a_pseudo 文件夹中的数据集，通过 embed.py 将数据向量化，生成 dataset_final 文件夹中的数据集
6. 读取 dataset_final 文件夹中的数据集，通过 get_embed_review_for_mmd.py 生成用于mmd处理的数据集，存放于 dataset_embed_review 文件夹中