# -*- coding: utf-8 -*-
import nltk, re, pprint

# 7.1 信息提取
# 信息提取结构
"""
使用句子分割器将该文档的原始文本分割成句，使用分词器将每个句子进一步细分为词。接下来，对每个句子进行词性标注，
在下一步命名实体识别中将证明这是非常有益的。在这一步，我们寻找每个句子中提到的潜在的有趣的实体。最后，我们
使用关系识别搜索文本中不同实体间的可能关系
要执行前面三项任务，我们可以定义一个函数，简单地连接 NLTK 中默认的句子分割
器1，分词器2和词性标注器3：
"""


def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]

"""
命名实体识别中，我们分割和标注可能组成一个有趣关系的实体。通常情况下，
这些将被定义为名词短语，例如 the knights who say "ni"或者适当的名称如 Monty Python。
在一些任务中，同时考虑不明确的名词或名词块也是有用的，如 every student 或 cats，这些
不必要一定与定义 NP 和适当名称一样的方式指示实体。
最后，在提取关系时，我们搜索对文本中出现在附近的实体对之间的特殊模式，并使用
这些模式建立元组记录实体之间的关系
"""
