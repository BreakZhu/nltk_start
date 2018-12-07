# -*- coding: utf-8 -*-
import nltk

# text = nltk.word_tokenize("And now for something completely different")
# print nltk.pos_tag(text)
# text = nltk.word_tokenize("They refuse to permit us to obtain the refuse permit")
# print nltk.pos_tag(text)
# text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
# print text.similar('woman')
# print text.similar('bought')
# print text.similar('over')
# print text.similar('the')
# tagged_token = nltk.tag.str2tuple('fly/NN')
# print tagged_token
sent = '''The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN other/AP topics/NNS ,/, 
        AMONG/IN them/PPO the/AT Atlanta/NP and/CC 
        Fulton/NP-tl County/NN-tl purchasing/VBG departments/NNS which/WDT it/PP
        said/VBD ``/`` ARE/BER well/QL operated/VBN and/CC follow/VB generally/R
        accepted/VBN practices/NNS which/WDT inure/VB to/IN the/AT best/JJT
        interest/NN of/IN both/ABX governments/NNS ''/'' ./.
        '''
# print [nltk.tag.str2tuple(t) for t in sent.split()]
# print nltk.corpus.brown.tagged_words()[:10]
# print nltk.corpus.brown.tagged_words(tagset='universal')[:20]
print nltk.corpus.sinica_treebank.tagged_words(tagset='universal')  # 汉语
print nltk.corpus.indian.tagged_words()  # 印地语
print nltk.corpus.mac_morpho.tagged_words()  # 葡萄牙语
print nltk.corpus.conll2002.tagged_words()  # 荷兰
print nltk.corpus.cess_cat.tagged_words()  # 加泰罗尼亚

from nltk.corpus import brown

brown_news_tagged = brown.tagged_words(categories='news',tagset='universal')
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
print tag_fd.keys()[:20]
# tag_fd.plot(20, cumulative=True)

# 构建了一个双连词的标记部分的 FreqDist。
word_tag_pairs = nltk.bigrams(brown_news_tagged)
print list(nltk.FreqDist(a[1] for (a, b) in word_tag_pairs if b[1] == 'N'))

# 新闻文本中最常见的动词是什么？让我们按频率排序所有动词
wsj = nltk.corpus.treebank.tagged_words()
word_tag_fd = nltk.FreqDist(wsj)
print [word + "/" + tag for (word, tag) in word_tag_fd if tag.startswith('V')]

cfd1 = nltk.ConditionalFreqDist(wsj)
print cfd1['yield'].keys()
cfd2 = nltk.ConditionalFreqDist((tag, word) for (word, tag) in wsj)
print cfd2['VN'].keys()


# 看看一些它们周围的文字
# print [w for w in cfd1.conditions() if 'VD' in cfd1[w] and 'VN' in cfd1[w]]
# idx1 = wsj.index(('kicked', 'VD'))
# print wsj[idx1 - 4:idx1 + 1]
# idx2 = wsj.index(('kicked', 'VN'))
# print wsj[idx2 - 4:idx2 + 1]


#  找出最频繁的名词标记的程序

def findtags(tag_prefix, tagged_text):
    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text if tag.startswith(tag_prefix))
    return dict((tag, cfd[tag].keys()[:5]) for tag in cfd.conditions())


tagdict = findtags('NN', nltk.corpus.brown.tagged_words(categories='news'))
for tag in sorted(tagdict):
    print tag, tagdict[tag]

# 假设我们正在研究词 often，想看看它是如何在文本中使用的。我们可以试着看看跟在
# often 后面的词汇：
brown_learned_text = brown.words(categories='learned')
# nltk.bigrams  2元分词法
print sorted(set(b for (a, b) in nltk.bigrams(brown_learned_text) if a == 'often'))
# 它使用 tagged_words()方法查看跟随词的词性标记可能更有指导性
brown_lrnd_tagged = brown.tagged_words(categories='learned', tagset='universal')
tags = [b[1] for (a, b) in nltk.bigrams(brown_lrnd_tagged) if a[0] == 'often']
fd = nltk.FreqDist(tags)
fd.tabulate()


# 我们考虑句子中的每个三词窗口，检查它们是否符合我们的标准。如果标记匹配，我们输出对应的词
def process(sentence):
    for (w1, t1), (w2, t2), (w3, t3) in nltk.trigrams(sentence):  # nltk trigrams 三元分词窗口
        if t1.startswith('V') and t2 == 'TO' and t3.startswith('V'):
            print w1, w2, w3


for tagged_sent in brown.tagged_sents():
    process(tagged_sent)

# 最后，让我们看看与它们的标记关系高度模糊不清的词。
# 了解为什么要标注这样的词是因为它们各自的上下文可以帮助我们弄清楚标记之间的区别
brown_news_tagged = brown.tagged_words(categories='news')
data = nltk.ConditionalFreqDist((word.lower(), tag) for (word, tag) in brown_news_tagged)
for word in data.conditions():
    if len(data[word]) > 3:
        tags = data[word].keys()
        print word, ' '.join(tags)
