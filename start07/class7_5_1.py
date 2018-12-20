#  -*- coding: utf-8  -*-
import nltk

# 7.5 命名实体识别
"""
我们简要介绍了命名实体（NEs）。命名实体是确切的名词短语，指示特定类型的个体，如组织、人、日期等。
表 7-3 列出了一些较常用的 NEs 类型。这些应该是不言自明的，除了“FACILITY”：
建筑和土木工程领域的人造产品；以及“GPE”：地缘政治实体，如城市、州/省、国家
组织 Georgia-Pacific Corp., WHO
人 Eddy Bonte, President Obama
地点 Murray River, Mount Everest
日期 June, 2008-06-29
时间 two fifty a m, 1:30 p.m.
货币 175 million Canadian Dollars, GBP 10.40
百分数 twenty pct, 18.75 %
设施 Washington Monument, Stonehenge
地缘政治实体 South East Asia, Midlothian

命名实体识别（NER）系统的目标是识别所有文字提及的命名实体。可以分解成两个子任务：确定 NE 的边界和确定其类型。
命名实体识别经常是信息提取中关系识别的前奏，它也有助于其他任务。例如：在问答系统（QA）中，我们试图提高信息检索的精确度，
不是返回整个页面而只是包含用户问题的答案的那些部分。大多数 QA 系统利用标准信息检索返回的文件，然后尝试分离文档中包含
答案的最小的文本片段。现在假设问题是 Who was the first President of the US？被检索的一个文档中包含下面这段话：
(5) The Washington Monument is the most prominent structure in Washington,
D.C. and one of the city’s early attractions. It was built in honor of George
Washington, who led the country to independence and then became its first
President.
分析问题时我们想到答案应该是 X was the first President of the US 的形式，其中 X
不仅是一个名词短语也是一个 PER 类型的命名实体。这应该使我们忽略段落中的第一句话。
虽然它包含 Washington 的两个出现，命名实体识别应该告诉我们：它们都不是正确的类型。
我们如何识别命名实体呢？一个办法是查找一个适当的名称列表。例如：识别地点时，我们可以使用地名辞典，
如亚历山大地名辞典或盖蒂地名辞典
地点检测，通过在新闻故事中简单的查找：查找地名辞典中的每个词是容易出错的；
案例区分可能有所帮助，但它们不是总会有的。
请看地名辞典很好的覆盖了很多国家的地点，却错误地认为 Sanchez 在多米尼加共和国
而 On 在越南。当然，我们可以从地名辞典中忽略这些地名，但这样一来当它们出现在一个
文档中时，我们将无法识别它们。
"""
"""
命名实体识别是一个非常适合用基于分类器类型的方法来处理的任务，这些方法我们在
名词短语分块时看到过。特别是，我们可以建立一个标注器，为使用 IOB 格式的每个块都
加了适当类型标签的句子中的每个词加标签。这里是 CONLL 2002（conll2002）荷兰语训练
数据的一部分：
Eddy N B-PER
Bonte N I-PER
is V O
woordvoerder N O
van Prep O
diezelfde Pron O
Hogeschool N B-ORG
. Punc O
在上面的表示中，每个标识符一行，与它的词性标记及命名实体标记一起。基于这个训
练语料，我们可以构造一个可以用来标注新句子的标注器，使用 nltk.chunk.conlltags2t
ree()函数将标记序列转换成一个块树。
NLTK 提供了一个已经训练好的可以识别命名实体的分类器，使用函数 nltk.ne_chun
k()访问。如果我们设置参数 binary=True，那么命名实体只被标注为 NE；否则，分类
器会添加类型标签，如 PERSON, ORGANIZATION, and GPE。
"""


def traverse(t):
    try:
        t.label()
    except AttributeError:
        print t,
    else:
        print '(', t.label(),
        for child in t:
            traverse(child)
        print ')'


sent = nltk.corpus.treebank.tagged_sents()[22]
t = nltk.ne_chunk(sent, binary=True)
traverse(t)
