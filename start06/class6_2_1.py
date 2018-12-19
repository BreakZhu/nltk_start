# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import names
import random

# 句子分割可以看作是一个标点符号的分类任务：每当我们遇到一个可能会结束一个句子
# 的符号，如句号或问号，我们必须决定它是否终止了当前句子。
# 第一步是获得一些已被分割成句子的数据，将它转换成一种适合提取特征的形式

sents = nltk.corpus.treebank_raw.sents()
tokens = []
boundaries = set()
offset = 0

# tokens 是单独句子标识符的合并链表，boundaries 是一个包含所有句子边界标识符索引的集合
for sent in nltk.corpus.treebank_raw.sents():
    tokens.extend(sent)  # 函数用于在列表末尾一次性追加另一个序列中的多个值
    offset += len(sent)
    boundaries.add(offset - 1)  # 每次存储句子位置


def punct_features(tokens, i):
    """
    我们需要指定用于决定标点是否表示句子边界的数据特征
    :param tokens:
    :param i:               判断下一句开始是否为大写，前一句结束为小写， 前一个词是否为单字母
    :return:
    """
    return {'next-word-capitalized': tokens[i + 1][0].isupper(), 'prevword': tokens[i - 1].lower(), 'punct': tokens[i],
            'prev-word-is-one-char': len(tokens[i - 1]) == 1}


# 基于这一特征提取器，我们可以通过选择所有的标点符号创建一个加标签的特征集的链表，然后标注它们是否是边界标识符
# 通过判断是否是 boundaries 确定是否是边界标识符

featuresets = [(punct_features(tokens, i), (i in boundaries)) for i in range(1, len(tokens) - 1) if tokens[i] in '.?!']
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print nltk.classify.accuracy(classifier, test_set)
print classifier.labels()


# 例 6-6. 基于分类的断句器

def segment_sentences(words):
    start = 0
    sents = []
    for i, word in words:
        if word in '.?!' and classifier.classify(words, i) == True:
            sents.append(words[start:i + 1])
            start = i + 1
    if start < len(words):
        sents.append(words[start:])


# 识别对话行为类型
"""
NPS 聊天语料库
包括超过 10,000 个来自即时消息会话的帖子。这些帖子都已经被贴上 15 种对话行为类型中的一种标签，
例如：“陈述”，“情感”，“yn 问
题”，“Continuer”。因此，我们可以利用这些数据建立一个分类器，识别新的即时消息帖子
的对话行为类型。第一步是提取基本的消息数据。我们将调用 xml_posts()来得到一个数
据结构，表示每个帖子的 XML 注释
"""
posts = nltk.corpus.nps_chat.xml_posts()[:10000]


# 下一步，我们将定义一个简单的特征提取器，检查帖子包含什么词：
def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains(%s)' % word.lower()] = True
    return features


"""
通过为每个帖子提取特征（使用 post.get('class') 获得一个帖子的对话行
为类型）构造训练和测试数据，并创建一个新的分类器
"""
featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print nltk.classify.accuracy(classifier, test_set)

# 识别文字蕴含
"""
应当强调，文字和假设之间的关系并不一定是逻辑蕴涵，而是一个人是否会得出结论：文本提供了合理的证据证明假设是真实的。
我们可以把 RTE 当作一个分类任务，尝试为每一对预测真/假标签。虽然这项任务的成功做法似乎看上去涉及语法分析、
语义和现实世界的知识的组合，RTE 的许多早期的尝试使用粗浅的分析基于文字和假设之间的在词级别的相似性取得了相当不错的结果。
在理想情况下，我们希望如果有一个蕴涵那么假设所表示的所有信息也应该在文本中表示。相反，如果假设中有的资料文本中没有，
那么就没有蕴涵。在我们的 RTE 特征探测器（例 6-7）中，我们让词（即词类型）作为信息的代理，我们的特征计数词重叠的程度和
假设中有而文本中没有的词的程度（由 hyp_extra()方法获取）。不是所有的词都是同样重要的——命名实体，
如人、组织和地方的名称，可能会更为重要，这促使我们分别为 words 和 nes（命名实体）提取不同的信息。
此外，一些高频虚词作为“停用词”被过滤掉。
图 6-7. “认识文字蕴涵”的特征提取器。RTEFeatureExtractor 类建立了一个除去一些停用词后在文本和假设中都有的词汇包，
然后计算重叠和差异
"""


def rte_features(rtepair):
    extractor = nltk.RTEFeatureExtractor(rtepair)   # 词的覆盖                                  # 假设中有儿实际没有
    features = {'word_overlap': len(extractor.overlap('word')), 'word_hyp_extra': len(extractor.hyp_extra('word')),
                'ne_overlap': len(extractor.overlap('ne')), 'ne_hyp_extra': len(extractor.hyp_extra('ne'))}
    return features
# 为了说明这些特征的内容，我们检查前面显示的文本/假设对 34 的一些属性
rtepair = nltk.corpus.rte.pairs(['rte3_dev.xml'])[33]
extractor = nltk.RTEFeatureExtractor(rtepair)
print extractor.text_words
print extractor.hyp_words
print extractor.overlap('word')
print extractor.overlap('ne')
print extractor.hyp_extra('word')

