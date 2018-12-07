# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import brown

#  N-gram 标注
# 一元标注（Unigram Unigram Unigram Unigram Tagging Tagging Tagging Tagging）
"""
一元标注器基于一个简单的统计算法：对每个标识符分配这个独特的标识符最有可能的
标记。例如：它将分配标记 JJ 给词 frequent 的所有出现，因为 frequent 用作一个形容词（例
如：a frequent word）比用作一个动词（例如：I frequent this cafe）更常见。一个一元标
注器的行为就像一个查找标注器（5.4 节），除了有一个更方便的建立它的技术，称为训练。
在下面的代码例子中，我们训练一个一元标注器，用它来标注一个句子，然后评估：
我们训练一个 UnigramTagger，通过在我们初始化标注器时指定已标注的句子数据作
为参数。训练过程中涉及检查每个词的标记，将所有词的最可能的标记存储在一个字典里面
这个字典存储在标注器内部
"""
from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
unigram_tagger.tag(brown_sents[2007])
print unigram_tagger.evaluate(brown_tagged_sents)

# 分离训练和测试数据
"""
在前面的例子,一个只是记忆它的训练数据，而不试图建立一个一般的模型的标
注器会得到一个完美的得分，但在标注新的文本时将是无用的。
相反，我们应该分割数据，90％为测试数据，其余 10％为测试数据
，90％为测试数据，其余 10％为测试数据
"""
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
print unigram_tagger.evaluate(test_sents)

# 一般的 N-gram 的标注
"""
一个 n-gram 标注器是一个 unigram 标注器的一般化，它的上下文是当前词和它前面 n1
个标识符的词性标记
1-gram标注器是一元标注器（unigram tagger）另一个名称：即用于标注
一个标识符的上下文的只是标识符本身。2-gram 标注器也称为二元标注器
（bigram taggers），3-gram 标注器也称为三元标注器（trigram taggers）
NgramTagger 类使用一个已标注的训练语料库来确定对每个上下文哪个词性标记最
有可能。在这里，我们看到一个 n-gram 标注器的特殊情况，即一个 bigram 标注器。首先，
我们训练它，然后用它来标注未标注的句子
"""
bigram_tagger = nltk.BigramTagger(train_sents)
print bigram_tagger.tag(brown_sents[2007])
unseen_sent = brown_sents[4203]
print bigram_tagger.tag(unseen_sent)
print bigram_tagger.evaluate(test_sents)

# 组合标注器
"""
    解决精度和覆盖范围之间的权衡的一个办法是尽可能的使用更精确的算法，但却在很多
    时候落后于具有更广覆盖范围的算法。例如：我们可以按如下方式组合 bigram 标注器、uni
    gram 标注器和一个默认标注器：
    1. 尝试使用 bigram 标注器标注标识符。
    2. 如果 bigram 标注器无法找到一个标记，尝试 unigram 标注器。
    3. 如果 unigram 标注器也无法找到一个标记，使用默认标注器
    将会丢弃那些只看到一次或两次的上下文
    nltk.BigramTagger(sents, cutoff=2, backoff=t1)
"""
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
print t2.evaluate(test_sents)

# 标注生词
"""
我们标注生词的方法仍然是回退到一个正则表达式标注器或一个默认标注器。这些都无
法利用上下文。因此，如果我们的标注器遇到词 blog，训练过程中没有看到过，它会分配相
同的标记，不论这个词出现的上下文是 the blog 还是 to blog。我们怎样才能更好地处理这
些生词，或词汇表以外的项目？
一个有用的基于上下文标注生词的方法是限制一个标注器的词汇表为最频繁的 n 个词，
使用 5.3 节中的方法替代每个其他的词为一个特殊的词 UNK。训练时，一个 unigram 标注器
可能会学到 UNK 通常是一个名词。然而，n-gram 标注器会检测它的一些其他标记中的上下
文。例如：如果前面的词是 to（标注为 TO），那么 UNK 可能会被标注为一个动词
"""
# 存储标注器

"""
在大语料库上训练一个标注器可能需要大量的时间。没有必要在每次我们需要的时候训
练一个标注器，很容易将一个训练好的标注器保存到一个文件以后重复使用。
"""
from cPickle import dump

output = open('t2.pkl', 'wb')
dump(t2, output, -1)
output.close()
# 现在，我们可以在一个单独的 Python 进程中载入我们保存的标注器：
from cPickle import load

input = open('t2.pkl', 'rb')
tagger = load(input)
input.close()
text = """The board's action shows what free enterprise is up against in our complex maze of regulatory laws ."""
tokens = text.split()
print tagger.tag(tokens)

# 性能限制
# 考虑一个trigram它遇到多少词性歧义的情况 我们可以根据经验决定这个问题的答案
"""
1/20 的 trigrams 是有歧义的 。给定当前单词及其前两个标记，根据训练数据，在
5％的情况中，有一个以上的标记可能合理地分配给当前词。假设我们总是挑选在这种含糊
不清的上下文中最有可能的标记，可以得出 trigram 标注器性能的一个下界
"""
cfd = nltk.ConditionalFreqDist(((x[1], y[1], z[0]), z[1]) for sent in brown_tagged_sents for x, y, z in nltk.trigrams(sent))
ambiguous_contexts = [c for c in cfd.conditions() if len(cfd[c]) > 1]
print sum(cfd[c].N() for c in ambiguous_contexts)*1.0 / cfd.N()

"""
调查标注器性能的另一种方法是研究它的错误。有些标记可能会比别的更难分配，可能
需要专门对这些数据进行预处理或后处理。一个方便的方式查看标注错误是混淆矩阵。它用
图表表示期望的标记（黄金标准）与实际由标注器产生的标记
"""
test_tags = [tag for sent in brown.sents(categories='editorial')
for (word, tag) in t2.tag(sent)]
gold_tags = [tag for (word, tag) in brown.tagged_words(categories='editorial')]
print nltk.ConfusionMatrix(gold_tags, test_tags)

# 跨句子边界标注
# 句子层面的 N-gram 标注
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
print t2.evaluate(test_sents)