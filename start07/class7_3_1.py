#  -*- coding: utf-8  -*-
import nltk
from nltk.classify import megam
"""
现在你对分块的作用有了一些了解，但我们并没有解释如何评估分块器。和往常一样，这需要一个合适的已标注语料库。
我们一开始寻找将 IOB 格式转换成 NLTK 树的机制，然后是使用已分块的语料库如何在一个更大的规模上做这个。
我们将看到如何为一个分块器相对一个语料库的准确性打分，再看看一些数据驱动方式搜索 NP 块。我们整个的重点在于扩
展一个分块器的覆盖范围
"""
# 读取 IOB 格式与 CoNLL2000 CoNLL2000 CoNLL2000 CoNLL2000 分块语料库
"""
使用 corpora 模块，我们可以加载已标注的《华尔街日报》文本，然后使用 IOB 符号分块。这个语料库提供的块类型有 NP，VP 和 PP。
正如我们已经看到的，每个句子使用多行表示，如下所示：
he PRP B-NP
accepted VBD B-VP
the DT B-NP
position NN I-NP
...
转换函数 chunk.conllstr2tree()用这些多行字符串建立一个树表示。此外，它允许我们选择使用三个块类型的任何子集，这里只是NP块
"""
text = '''
he PRP B-NP
accepted VBD B-VP
the DT B-NP
position NN I-NP
of IN B-PP
vice NN B-NP
chairman NN I-NP
of IN B-PP
Carlyle NNP B-NP
Group NNP I-NP
, , O
a DT B-NP
merchant NN I-NP
banking NN I-NP
concern NN I-NP
. . O
'''
nltk.chunk.conllstr2tree(text, chunk_types=['NP']).draw()

"""
我们可以使用 NLTK 的 corpus 模块访问较大量的已分块文本。CoNLL2000 分块语料
库包含 27 万词的《华尔街日报文本》，分为“训练”和“测试”两部分，标注有词性标记和
IOB 格式分块标记。我们可以使用 nltk.corpus.conll2000 访问这些数据。下面是一个读
取语料库的“训练”部分的 100 个句子的例子：
"""
from nltk.corpus import conll2000

print conll2000.chunked_sents('train.txt')[99]
print conll2000.chunked_sents('train.txt', chunk_types=['NP'])[99]

# 简单评估和基准
"""
我们可以访问一个已分块语料，可以评估分块器。我们开始为琐碎的不创建任何块的块分析器 cp 建立一个基准（baseline）：
"""

from nltk.corpus import conll2000

# cp = nltk.RegexpParser("")  #
cp = nltk.RegexpParser(r"NP: {<[CDJNP].*>+}")
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
print cp.evaluate(test_sents)

"""
正如你看到的，这种方法达到相当好的结果。但是，我们可以采用更多数据驱动的方法
改善它，在这里我们使用训练语料找到对每个词性标记最有可能的块标记（I、O 或 B）。换
句话说，我们可以使用 unigram 标注器（5.4 节）建立一个分块器。但不是尝试确定每个词
的正确的词性标记，而是给定每个词的词性标记，尝试确定正确的块标记
我们定义了 UnigramChunker 类，使用 unigram 标注器给句子加块标记。
这个类的大部分代码只是用来在 NLTK 的 ChunkParserI 接口使用的分块树表示和嵌入式
标注器使用的 IOB 表示之间镜像转换。类定义了两个方法：一个构造函数，当我们建立
一个新的 UnigramChunker 时调用；一个 parse 方法，用来给新句子分块。
例 7-4. 使用 unigram 标注器对名词短语分块
"""


class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        """
        需要训练句子的一个链表，这将是块树的形式。它首先将训练数据转换成适
        合训练标注器的形式，使用 tree2conlltags 映射每个块树到一个词，标记，块三元组的链
        表。然后使用转换好的训练数据训练一个 unigram 标注器，并存储在 self.tagger 供以后使用
        :param train_sents:
        """
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]
        # self.tagger = nltk.UnigramTagger(train_data)     # 修改为BigramTagger 就变成了
        self.tagger = nltk.BigramTagger(train_data)

    def parse(self, sentence):
        """
        parse 方法取一个已标注的句子作为其输入，以从那句话提取词性标记开始。然后使
        用在构造函数中训练过的标注器 self.tagger，为词性标记标注 IOB 块标记。接下来，提取
        块标记，与原句组合，产生 conlltags。最后，使用 conlltags2tree 将结果转换成一个块树。
        现在我们有了 UnigramChunker，可以使用 CoNLL2000 分块语料库训练它，并测试其性能
        :param sentence:
        :return:
        """
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)


test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
unigram_chunker = UnigramChunker(train_sents)
print unigram_chunker.evaluate(test_sents)
"""
这个分块器相当不错，达到整体 F 度量 83％的得分。让我们来看一看通过使用 unigram 标注器分配一个标记给每个语料库中出现的
词性标记，它学到了什么：
"""
postags = sorted(set(pos for sent in train_sents for (word, pos) in sent.leaves()))
print unigram_chunker.tagger.tag(postags)

# 训练基于分类器的分块器
"""
基于分类器的 NP 分块器的基础代码如例 7-5 所示。它包括两个类：第一个类 1 几乎与
例 6-5 中 ConsecutivePosTagger 类相同。仅有的两个区别是它调用一个不同的特征提取
器 2，使用 MaxentClassifier 而不是 NaiveBayesClassifier 3 第二个类基本上是标注
器类的一个包装器，将它变成一个分块器。训练期间，这第二个类映射训练语料中的块树到
标记序列；在 parse()方法中，它将标注器提供的标记序列转换回一个块树。
例 7-5. 使用连续分类器对名词短语分块
"""


class ConsecutiveNPChunkTagger(nltk.TaggerI):
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
        history = []
        for i, (word, tag) in enumerate(tagged_sent):
            featureset = npchunk_features(untagged_sent, i, history)
            train_set.append((featureset, tag))
            history.append(tag)
            self.classifier = nltk.MaxentClassifier.train(train_set, nltk.classify.MaxentClassifier.ALGORITHMS[2],
                                                          trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
        tag = self.classifier.classify(featureset)
        history.append(tag)
        return zip(sentence, history)


class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w, t), c) for (w, t, c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w, t, c) for ((w, t), c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)


# def npchunk_features(sentence, i, history):
#     word, pos = sentence[i]
#     if i == 0:
#         prevword, prevpos = "<START>", "<START>"
#     else:
#         prevword, prevpos = sentence[i - 1]
#     # return {"pos": pos, "prevpos": prevpos}
#     return {"pos": pos, "word": word, "prevpos": prevpos}  # 把前词作为特征


"""
我们尝试用多种附加特征扩展特征提取器，例如：预取特征、配对功能
和复杂的语境特征。这最后一个特征，被称为 tags-since-dt，创建一个字符串，描
述自最近的限定词以来遇到的所有词性标记
"""


def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
    else:
        prevword, prevpos = sentence[i - 1]
    if i == len(sentence) - 1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i + 1]
    return {"pos": pos,
            "word": word,
            "prevpos": prevpos,
            "nextpos": nextpos,
            "prevpos+pos": "%s+%s" % (prevpos, pos),
            "pos+nextpos": "%s+%s" % (pos, nextpos),
            "tags-since-dt": tags_since_dt(sentence, i)}


def tags_since_dt(sentence, i):
    tags = set()
    for word, pos in sentence[:i]:
        if pos == 'DT':
            tags = set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))


chunker = ConsecutiveNPChunker(train_sents)
print chunker.evaluate(test_sents)
