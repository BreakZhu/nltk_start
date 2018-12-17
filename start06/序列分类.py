# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import names, brown
import random

"""
为了捕捉相关的分类任务之间的依赖关系，我们可以使用联合分类器模型，收集有关输
入，选择适当的标签。在词性标注的例子中，各种不同的序列分类器模型可以被用来为一个
给定的句子中的所有的词共同选择词性标签。
一种序列分类器策略，称为连续分类或贪婪序列分类，是为第一个输入找到最有可能的
类标签，然后使用这个问题的答案帮助找到下一个输入的最佳的标签。这个过程可以不断重
复直到所有的输入都被贴上标签。这是被 5.5 节的 bigram 标注器采用的方法，它一开始为
215
句子的第一个词选择词性标记，然后为每个随后的词选择标记，基于词本身和前面词的预测
的标记。
在例 6-5 演示了这一策略。首先，我们必须扩展我们的特征提取函数使其具有参数 his
tory，它提供一个我们到目前为止已经为句子预测的标记的链表�。history 中的每个标记
对应句子中的一个词。但是请注意，history 将只包含我们已经归类的词的标记，也就是目
标词左侧的词。因此，虽然是有可能查看目标词右边的词的某些特征，但查看那些词的标记
是不可能的（因为我们还未产生它们）。
已经定义了特征提取器，我们可以继续建立我们的序列分类器�。在训练中，我们使用
已标注的标记为征提取器提供适当的历史信息，但标注新的句子时，我们基于标注器本身的
输出产生历史信息。
例 6-5. 使用连续分类器进行词性标注。
"""


def pos_features(sentence, i, history):
    features = {"suffix(1)": sentence[i][-1:],
                "suffix(2)": sentence[i][-2:],
                "suffix(3)": sentence[i][-3:]}
    if i == 0:
        features["prev-word"] = "<START>"
        features["prev-tag"] = "<START>"
    else:
        features["prev-word"] = sentence[i - 1]
        features["prev-tag"] = history[i - 1]
    return features


class ConsecutivePosTagger(nltk.TaggerI):
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = pos_features(untagged_sent, i, history)
                train_set.append((featureset, tag))
                history.append(tag)
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = pos_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

tagged_sents = brown.tagged_sents(categories='news')
size = int(len(tagged_sents) * 0.1)
train_sents, test_sents = tagged_sents[size:], tagged_sents[:size]
tagger = ConsecutivePosTagger(train_sents)
print tagger.evaluate(test_sents)
