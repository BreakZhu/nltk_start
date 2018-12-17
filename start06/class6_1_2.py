# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import names
import random

"""
我们看到了语料库的几个例子，那里文档已经按类别标记。使用这些语料库，我们可以建立分类器，自动给新文档添加适当的类别标签。
首先，我们构造一个标记了相应类别的文档清单。
"""
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

"""我们为文档定义一个特征提取器，这样分类器就会知道哪些方面的数据应注意
（见例 6-2）。对于文档主题识别，我们可以为每个词定义一个特性表示该文档是否包含这
个词。为了限制分类器需要处理的特征的数目，我们一开始构建一个整个语料库中前 2000
个最频繁词的链表。然后，定义一个特征提取器，简单地检查这些词是否在一个给定的文档中。
例 6-2. 一个文档分类的特征提取器，其特征表示每个词是否在一个给定的文档中
"""
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())  # 获取影评中所有数据
word_featrues = all_words.keys()[:2000]


def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_featrues:
        features['contains(%s)' % word] = (word in document_words)
    return features


print document_features(movie_reviews.words('pos/cv957_8737.txt'))

"""
训练和测试一个分类器进行文档分类
"""
featrueset = [(document_features(d), c) for (d, c) in documents]
train_set, test_set = featrueset[200:], featrueset[:200]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print nltk.classify.accuracy(classifier, train_set)
classifier.show_most_informative_features(5)

from nltk.corpus import brown

"""
我们建立了一个正则表达式标注器，通过查找词内部的组成，为词选择词性
标记。然而，这个正则表达式标注器是手工制作的。作为替代，我们可以训练一个分类器来
算出哪个后缀最有信息量。首先，让我们找出最常见的后缀
"""
suffix_fdist = nltk.FreqDist()
for word in brown.words():
    word = word.lower()
    suffix_fdist[word[-1:]] += 1
    suffix_fdist[word[-2:]] += 1
    suffix_fdist[word[-3:]] += 1
common_suffixes = suffix_fdist.keys()[:100]
print common_suffixes


def pos_features(word):
    """
    定义一个特征提取器函数，检查给定的单词的这些后缀
    :param word:
    :return:
    """
    features = {}
    for suffix in common_suffixes:
        features['endswith(%s)' % suffix] = word.lower().endswith(suffix)
    return features


"""
特征提取函数的行为就像有色眼镜一样，强调我们的数据中的某些属性（颜色），并使
其无法看到其他属性。分类器在决定如何标记输入时，将完全依赖它们强调的属性。在这种
情况下，分类器将只基于一个给定的词拥有（如果有）哪个常见后缀的信息来做决定。
现在，我们已经定义了我们的特征提取器，可以用它来训练一个新的“决策树”的分类
器
"""
# 使用决策树
tagged_word = brown.tagged_words(categories="news")
featuresets = [(pos_features(n), g) for (n, g) in tagged_word]
print len(featuresets)
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.DecisionTreeClassifier.train(train_set)
print nltk.classify.accuracy(classifier, test_set)
print classifier.classify(pos_features('cats'))
# 输出树的层级
print classifier.pseudocode(depth=4)

"""
例 6-4. 一个词性分类器，它的特征检测器检查一个词出现的上下文以便决定应该分配
的词性标记。特别的，前面的词被作为一个特征。
"""


def pos_features(sentence, i):
    features = {"suffix(1)": sentence[i][-1:],
                "suffix(2)": sentence[i][-2:],
                "suffix(3)": sentence[i][-3:]}
    if i == 0:
        features["prev-word"] = "<START>"
    else:
        features["prev-word"] = sentence[i - 1]
    return features

pos_features(brown.sents()[0], 8)
tagged_sents = brown.tagged_sents(categories='news')
featuresets = []
for tagged_sent in tagged_sents:
    untagged_sent = nltk.tag.untag(tagged_sent)
    for i, (word, tag) in enumerate(tagged_sent):
        featuresets.append((pos_features(untagged_sent, i), tag))
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)

