# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import brown

"""
n-gram 标注器从前面的上下文中获得的唯一的信息是标
记，虽然词本身可能是一个有用的信息源。n-gram 模型使用上下文中的词的其他特征为条
件是不切实际的。在本节中，我们考察 Brill 标注，一种归纳标注方法，它的性能很好，使
用的模型只有 n-gram 标注器的很小一部分
"""
print nltk.tag.brill.nltkdemo18()