# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import names
import random


def gender_features(word):
    # return {'last_letter': word[-1]}
    return {'last_letter': word[-1].lower()
            # , "word_length": len(word)
            # , "first_letter": str(word[:1]).lower()
            # , "last_first_letters": str(word[-1]+word[:1]).lower()
            # , "first_last_letters": str(word[:1]+word[-1]).lower()
            }


def gender_features3(word):
    return {'suffix1': word[-1:], 'suffix2': word[-2:], 'perfix_1': word[:1], "perfix_2": word[:2]}


# print gender_features('Shrek')  # 通过最后的字母确定性别

names = (
    [(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
random.shuffle(names)

featuresets = [(gender_features(n), g) for (n, g) in names]
train_set, test_set = featuresets[800:], featuresets[:200]

"""
在这些情况下，使用函数 nltk.classify.apply_features，返回一个行为像一个链表而
不会在内存存储所有特征集的对象
"""
from nltk.classify import apply_features

train_set = apply_features(gender_features, names[500:])
test_set = apply_features(gender_features, names[:500])

classifier = nltk.NaiveBayesClassifier.train(train_set)
print classifier.classify(gender_features('John Cena'))
print classifier.classify(gender_features('Trinity'))
print nltk.classify.accuracy(classifier, test_set)
"""
此列表显示训练集中以 a 结尾的名字中女性是男性的 38 倍，而以 k 结尾名字中男性是
女性的 31 倍。这些比率称为似然比，可以用于比较不同特征-结果关系
"""
print classifier.show_most_informative_features(5)


# 例 6-1. 一个特征提取器，过拟合性别特征。这个特征提取器返回的特征集包括大量指定的特征，
# 从而导致对于相对较小的名字语料库过拟合
def gender_features2(name):
    features = {"firstletter": name[0].lower(), "lastletter": str(name[-1]).lower()}
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count(%s)" % letter] = name.lower().count(letter)
        features["has(%s)" % letter] = (letter in name.lower())
    return features


print gender_features2('John')
"""
你要用于一个给定的学习算法的特征的数目是有限的——如果你提供太多的特
征，那么该算法将高度依赖你的训练数据的特，性而一般化到新的例子的效果不会很好。这
个问题被称为过拟合
"""
featuresets = [(gender_features2(n), g) for (n, g) in names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print nltk.classify.accuracy(classifier, test_set)

"""
yn 结尾的名字显示以女性为主，尽管事实上，n 结尾的名字往往是男性；以 ch 结尾的名字通常
是男性，尽管以 h 结尾的名字倾向于是女性
"""
train_names = names[1500:]
devtest_names = names[500:1500]
test_names = names[:500]
train_set = [(gender_features3(n), g) for (n, g) in train_names]  # 训练集数据
devtest_set = [(gender_features3(n), g) for (n, g) in devtest_names]  # 开发测试集
test_set = [(gender_features3(n), g) for (n, g) in test_names]  # 测试
classifier = nltk.NaiveBayesClassifier.train(train_set)
print nltk.classify.accuracy(classifier, devtest_set)

errors = []
for (name, tag) in devtest_names:
    guess = classifier.classify(gender_features3(name))
    if guess != tag:
        errors.append((tag, guess, name))
for (tag, guess, name) in sorted(errors):  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    print 'correct=%-8s guess=%-8s name=%-30s' % (tag, guess, name)


