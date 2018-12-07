# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
nltk.FreqDist(tags).max()
# 现在我们可以创建一个将所有词都标注成 NN 的标注器
raw = 'I do not like green eggs and ham, I do not like them Sam I am!'
tokens = nltk.word_tokenize(raw)
default_tagger = nltk.DefaultTagger('NN')
print default_tagger.tag(tokens)
print default_tagger.evaluate(brown_tagged_sents)

# 正则表达式标注器

"""
我们可能会猜测任一以 ed
结尾的词都是动词过去分词，任一以's 结尾的词都是名词所有格。可以用一个正则表达式的
列表表示这些
"""
patterns = [(r'.*ing$', 'VBG'),  # gerunds
            (r'.*ed$', 'VBD'),  # simple past
            (r'.*es$', 'VBZ'),  # 3rd singular present
            (r'.*ould$', 'MD'),  # modals
            (r'.*\'s$', 'NN$'),  # possessive nouns
            (r'.*s$', 'NNS'),  # plural nouns
            (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
            (r'.*', 'NN')  # nouns (default)
            ]
regexp_tagger = nltk.RegexpTagger(patterns)
regexp_tagger.tag(brown_sents[3])
print regexp_tagger.evaluate(brown_tagged_sents)

# 查询标注器
"""
    很多高频词没有 NN 标记。让我们找出 100 个最频繁的词，存储它们最有可能的标记。
    然后我们可以使用这个信息作为“查找标注器”（NLTK UnigramTagger）的模型    
"""
fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = fd.keys()[:1000]
likely_tags = dict((word, cfd[word].max()) for word in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
print baseline_tagger.evaluate(brown_tagged_sents)
sent = brown.sents(categories='news')[3]
print baseline_tagger.tag(sent)


# 查找标注器的性能，使用不同大小的模型。
def performance(cfd, wordlist):
    lt = dict((word, cfd[word].max()) for word in wordlist)
    baseline_tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger('NN'))
    return baseline_tagger.evaluate(brown.tagged_sents(categories='news'))


def display():
    import pylab
    words_by_freq = list(nltk.FreqDist(brown.words(categories='news')))
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    sizes = 2 ** pylab.arange(16)
    perfs = [performance(cfd, words_by_freq[:size]) for size in sizes]
    pylab.plot(sizes, perfs, '-bo')
    pylab.title('Lookup Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()
"""
随着模型规模的增长，最初的性能增加迅速，最终达到一个稳定水平，这
时模型的规模大量增加性能的提高很小
"""
display()

# 评估
# 这些工具的性能评估是 NLP 的一个中心主题
