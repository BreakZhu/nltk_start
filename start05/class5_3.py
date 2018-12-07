# -*- coding: utf-8 -*-

import nltk

alice = nltk.corpus.gutenberg.words('carroll-alice.txt')
vocab = nltk.FreqDist(alice)
v1000 = list(vocab)[:1000]
mapping = nltk.defaultdict(lambda: 'UNK')
for v in v1000:
    mapping[v] = v
alice2 = [mapping[v] for v in alice]
print alice2[:100]
print len(set(alice2))

# 递增地更新字典，按值降序排序
counts = nltk.defaultdict(int)
from nltk.corpus import brown

for (word, tag) in brown.tagged_words(categories='news'):
    counts[tag] += 1
print counts['N']
print list(counts)
from operator import itemgetter

# 第二个参数使用函数 itemgetter()指定排序键。在一般情况下，itemgetter(n)
# 返回一个函数，这个函数可以在一些其他序列对象上被调用获得这个序列的第 n 个元素
# 的最后一个参数指定项目是否应被按相反的顺序返回，即频率值递减。
print sorted(counts.items(), key=itemgetter(1), reverse=True)
# 第二种处理方式 按它们最后两个字母索引词汇
last_letters = nltk.defaultdict(list)
words = nltk.corpus.words.words('en')
for word in words:
    key = word[-2:]
    last_letters[key].append(word)
print last_letters['ly']
# 下面的例子使用相同的模式创建一个颠倒顺序的词字典
anagrams = nltk.defaultdict(list)
for word in words:
    key = ''.join(sorted(word))
    anagrams[key].append(word)
print anagrams['aeilnrt']
# 由于积累这样的词是如此常用的任务，NLTK 以 nltk.Index()的形式提供一个创建 defaultdict(list)更方便的方式
anagrams = nltk.Index((''.join(sorted(w)), w) for w in words)
print anagrams['aeilnrt']

# 它的条目的默认值也是一个字典（其默认值是 int()，即 0）。请注意我们如何遍历已标注语料库的双连词，
# 每次遍历处理一个词-标记对每次通过循环时，我们更新字典 pos 中的条目 (t1, w2)，一个标记和它后面的词。当我们在 pos
# 中查找一个项目时，我们必须指定一个复合键，然后得到一个字典对象。一个 POS 标注
# 器可以使用这些信息来决定词 right，前面是一个限定词时，应标注为 ADJ。
pos = nltk.defaultdict(lambda: nltk.defaultdict(int))
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
for ((w1, t1), (w2, t2)) in nltk.bigrams(brown_news_tagged):
    pos[(t1, w2)][t2] += 1
print pos[('DET', 'right')]
