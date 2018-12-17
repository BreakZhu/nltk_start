# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import names
import random

# 句子分割可以看作是一个标点符号的分类任务：每当我们遇到一个可能会结束一个句子
# 的符号，如句号或问号，我们必须决定它是否终止了当前句子。
# 第一步是获得一些已被分割成句子的数据，将它转换成一种适合提取特征的形式

