#  -*- coding: utf-8  -*-
import nltk
import re
"""
一旦文本中的命名实体已被识别，我们就可以提取它们之间存在的关系。如前所述，我
们通常会寻找指定类型的命名实体之间的关系。进行这一任务的方法之一是首先寻找所有
(X, α, Y)形式的三元组，其中 X 和 Y 是指定类型的命名实体，α表示 X 和 Y 之间关系的
字符串。然后我们可以使用正则表达式从α的实体中抽出我们正在查找的关系
"""
IN = re.compile(r'.*\bin\b(?!\b.+ing)')
for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
    for rel in nltk.sem.extract_rels('ORG', 'LOC', doc, corpus='ieer', pattern=IN):
        print nltk.sem.rtuple(rel)

from nltk.corpus import conll2002
vnv = """
      (
      is/V| # 3rd sing present and
      was/V| # past forms of the verb zijn ('be')
      werd/V| # and also present
      wordt/V # past of worden ('become')
      )
      .* # followed by anything
      van/Prep # followed by van ('of')
      """
VAN = re.compile(vnv, re.VERBOSE)
for doc in conll2002.chunked_sents('ned.train'):
    for r in nltk.sem.extract_rels('PER', 'ORG', doc,corpus='conll2002', pattern=VAN):
      print nltk.sem.clause(r, relsym="VAN")

"""
7.7
信息提取系统搜索大量非结构化文本，寻找特定类型的实体和关系，并用它们来填充有
组织的数据库。这些数据库就可以用来寻找特定问题的答案。
信息提取系统的典型结构以断句开始，然后是分词和词性标注。接下来在产生的数据中
搜索特定类型的实体。最后，信息提取系统着眼于文本中提到的相互临近的实体，并试
图确定这些实体之间是否有指定的关系。
实体识别通常采用分块器，它分割多标识符序列，并用适当的实体类型给它们加标签。
常见的实体类型包括组织、人员、地点、日期、时间、货币、GPE（地缘政治实体）。
用基于规则的系统可以构建分块器，例如：NLTK 中提供的 RegexpParser 类；或使
用机器学习技术，如本章介绍的 ConsecutiveNPChunker。在这两种情况中，词性标
记往往是搜索块时的一个非常重要的特征。
虽然分块器专门用来建立相对平坦的数据结构，其中没有任何两个块允许重叠，但它们
可以被串联在一起，建立嵌套结构。关系抽取可以使用基于规则的系统，它通常查找文本中的连结实体和相关的词的特定模
式；或使用机器学习系统，通常尝试从训练语料自动学习这种模式
"""
