# -*- coding: utf-8 -*-
import nltk, re, pprint

#  7.2 分块
"""
我们将用于实体识别的基本技术是分块（chunking），分割和标注图 7-2 所示的多标识
符序列。小框显示词级标识符和词性标注，大框显示较高级别的分块。每个这种较大的框叫做一大块（chunk）。就像分词忽略空白符，
分块通常选择标识符的一个子集。同样像分词一样，分块构成的源文本中的片段不能重叠。
"""
# 名词短语分块
# NP-分块 OR NP-chunking
"""
NP-分块信息最有用的来源之一是词性标记。这是在我们的信息提取系统中进行词性标
注的动机之一。我们在例 7-1 中用一个已经标注词性的例句来演示这种方法。为了创建一个
NP-块，我们将首先定义一个块语法，规则句子应如何分块。在本例中，我们将用一个正则
表达式规则定义一个简单的语法。这条规则是说一个 NP-块由一个可选的限定词（DT）
后面跟着任何数目的形容词（JJ）然后是一个名词（NN）组成。使用此语法，我们创建了
一个块分析器1，测试我们的例句④。结果是一棵树，我们可以输出⑤或图形显示⑥。
"""
# 例 7-1. 一个简单的基于正则表达式的 NP 分块器的例子。
sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"), ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),
            ("the", "DT"), ("cat", "NN")]
grammar = "NP: {<DT>?<JJ>*<NN>}"
cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
print result
# result.draw()

# 标记模式
"""
组成一个块语法的规则使用标记模式来描述已标注的词的序列。一个标记模式是一个用尖括号分隔的词性标记序列，如<DT>?<JJ>*<NN>。
标记模式类似于正则表达式模式
another/DT sharp/JJ dive/NN
trade/NN figures/NNS
any/DT new/JJ policy/NN measures/NNS
earlier/JJR stages/NNS
Panamanian/JJ dictator/NN Manuel/NNP Noriega/NNP
我们可以使用轻微改进的上述第一个标记模式来匹配这些名词短语，如<DT>?<JJ.*
>*<NN.*>+。这将把任何以一个可选的限定词开头，后面跟零个或多个任何类型的形容
词（包括相对形容词，如 earlier/JJR），后面跟一个或多个任何类型的名词的标识符序列分
块。然而，很容易找到许多该规则不包括的更复杂的例子：
his/PRP$ Mansion/NNP House/NNP speech/NN
the/DT price/NN cutting/VBG
3/CD %/NN to/TO 4/CD %/NN
more/JJR than/IN 10/CD %/NN
the/DT fastest/JJS developing/VBG trends/NNS
's/POS skill/NN
"""

# 用正则表达式分块
# 例 7-2. 简单的名词短语分块器。
# chunk determiner/possessive, adjectives and nouns
# chunk sequences of proper nouns
grammar = r"""
NP: {<DT|PP\$>?<JJ>*<NN>} # chunk determiner/possessive, adjectives and nouns
{<NNP>+} # chunk sequences of proper nouns
"""
cp1 = nltk.RegexpParser(grammar)
sentence = [("Rapunzel", "NNP"), ("let", "VBD"), ("down", "RP"), ("her", "PP$"), ("long", "JJ"), ("golden", "JJ"),
            ("hair", "NN")]
print cp1.parse(sentence)

nouns = [("money", "NN"), ("market", "NN"), ("fund", "NN")]
grammar = "NP: {<NN><NN>} # Chunk two consecutive nouns"
cp2 = nltk.RegexpParser(grammar)
print cp2.parse(nouns)

# 探索文本语料库
"""
如何在已标注的语料库中提取匹配的特定的词性标记序列的短语
"""
# cp3 = nltk.RegexpParser('CHUNK: {<V.*> <TO> <V.*>}')
# brown = nltk.corpus.brown
# for sent in brown.tagged_sents():
#     tree = cp3.parse(sent)
#     for subtree in tree.subtrees():
#         if subtree.label() == 'CHUNK':  # 注意由于版本问题需要修改 subtree.node 为 subtree.label()
#             print subtree
"""
表 7-2. 三个加缝隙规则应用于同一个块。
          整个块                      块中间                 块结尾
输入 [a/DT little/JJ dog/NN]    [a/DT little/JJ dog/NN]  [a/DT little/JJ dog/NN]
操作   Chink “DT JJ NN”           Chink “JJ”              Chink “NN”
模式     }DT JJ NN{                    }JJ{                       }NN{
输出 a/DT little/JJ dog/NN     [a/DT] little/JJ [dog/NN]    [a/DT little/JJ] dog/NN
我们将整个句子作为一个块，然后练习加缝隙。
例 7-3. 简单的加缝器
"""
grammar = r"""
NP:
{<.*>+} # Chunk everything
}<VBD|IN>+{ # Chink sequences of VBD and IN
"""
sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),("dog", "NN"), ("barked", "VBD"),
            ("at", "IN"), ("the", "DT"), ("cat", "NN")]
cp = nltk.RegexpParser(grammar)
print cp.parse(sentence)
"""
块的表示：标记与树
IOB 标记已成为文件中表示块结构的标准方式，我们也将使用这种格式。下面是图 7-3
中的信息如何出现在一个文件中的：
We PRP B-NP
saw VBD O
the DT B-NP
little JJ I-NP
yellow JJ I-NP
dog NN I-NP
在此表示中，每个标识符一行，和它的词性标记与块标记一起。这种格式允许我们表示
多个块类型，只要块不重叠。正如我们前面所看到的，块的结构也可以使用树表示。这有利
于使每块作为一个组成部分可以直接操作
"""