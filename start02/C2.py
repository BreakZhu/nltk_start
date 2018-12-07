# -*- coding: utf-8 -*-

import nltk
from nltk.book import gutenberg  # 古腾堡预料
from nltk.corpus import webtext  # 网络小文本
from nltk.corpus import nps_chat  # 网络聊天室数据

for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))  # 给出原始文本字符数
    num_words = len(gutenberg.words(fileid))  # 告诉我们文本中出现的词汇个数包括词之间的空格
    num_sens = len(gutenberg.sents(fileid))  # 函数把文本划分成句子，其中每一个句子是一个词链表
    num_vocab = len(set([w.lower() for w in gutenberg.words(fileid)]))
    #      平均词长\平均句长\词语多样性
    print int(num_chars / num_words), int(num_words / num_sens), int(num_words / num_vocab), fileid

#  的网络文本小集合的内容包括 Firefox 交流论坛
for fileid in webtext.fileids():
    print fileid, webtext.raw(fileid)[:65], '...'

# 10-19-20s_706posts.xml 包含 2006 年 10 月 19 日从 20 多岁聊天室收集的 706 个帖子
chatroom = nps_chat.posts('10-19-20s_706posts.xml')
print chatroom[123]


from nltk.corpus import inaugural
cfd = nltk.ConditionalFreqDist(
        (target, fileid[:4])
        for fileid in inaugural.fileids()
        for w in inaugural.words(fileid)
        for target in ['america', 'citizen']
        if w.lower().startswith(target))
cfd.plot()
