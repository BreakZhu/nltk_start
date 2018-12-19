# -*- coding: utf-8 -*-

# 6.3 评估

# 测试集
import random
from nltk.corpus import brown

tagged_sents = list(brown.tagged_sents(categories='news'))
random.shuffle(tagged_sents)
size = int(len(tagged_sents) * 0.1)
train_set, test_set = tagged_sents[size:], tagged_sents[:size]
"""
在这种情况下，我们的测试集和训练集将是非常相似的。训练集和测试集均取自同一文
体，所以我们不能相信评估结果可以推广到其他文体。更糟糕的是，因为调用 random.shuffle()，
测试集中包含来自训练使用过的相同的文档的句子。如果文档中有相容的模式（也
就是说，如果一个给定的词与特定词性标记一起出现特别频繁），那么这种差异将体现在开
发集和测试集。一个稍好的做法是确保训练集和测试集来自不同的文件
"""
file_ids = brown.fileids(categories='news')
size = int(len(file_ids) * 0.1)
# train_set = brown.tagged_sents(file_ids[size:])
# test_set = brown.tagged_sents(file_ids[:size])
train_set = brown.tagged_sents(categories='news')
test_set = brown.tagged_sents(categories='fiction')

# 准确度
"""
用于评估一个分类最简单的度量是准确度，测量测试集上分类器正确标注的输入的比
例。例如：一个名字性别分类器，在包含 80 个名字的测试集上预测正确的名字有 60 个，它
有 60/80= 75％的准确度。nltk.classify.accuracy()函数会在给定的测试集上计算分类器
模型的准确度
"""
# classifier = nltk.NaiveBayesClassifier.train(train_set)
# print 'Accuracy: %4.2f' % nltk.classify.accuracy(classifier, test_set)

# 精确度和召回率
"""
• 真阳性是相关项目中我们正确识别为相关的。
• 真阴性是不相关项目中我们正确识别为不相关的。
• 假阳性（或 I 型错误）是不相关项目中我们错误识别为相关的。
• 假阴性（或 II 型错误）是相关项目中我们错误识别为不相关的
"""
"""
精确度（Precision），表示我们发现的项目中有多少是相关的，TP/(TP+ FP)。
召回率（Recall），表示相关的项目中我们发现了多少，TP/(TP+ FN)。
F-度量值（F-Measure）（或 F-得分，F-Score），组合精确度和召回率为一个单独的得分，
被定义为精确度和召回率的调和平均数(2 × Precision × Recall)/(Precision+Recall)
"""

# 混淆矩阵
"""
当处理有 3 个或更多的标签的分类任务时，基于模型错误类型细分模型的错误是有信息
量的。一个混淆矩阵是一个表，其中每个 cells[i,j]表示正确的标签 i 被预测为标签 j 的次数。
因此，对角线项目（即 cells[i,i]）表示正确预测的标签，非对角线项目表示错误。在下面的
例子中，我们为 5.4 节中开发的 unigram 标注器生成一个混淆矩阵
"""

# 交叉验证
"""
为了评估我们的模型，我们必须为测试集保留一部分已标注的数据。正如我们已经提到，
如果测试集是太小了，我们的评价可能不准确。然而，测试集设置较大通常意味着训练集设
置较小，如果已标注数据的数量有限，这样设置对性能会产生重大影响。
这个问题的解决方案之一是在不同的测试集上执行多个评估，然后组合这些评估的得
分，这种技术被称为交叉验证。特别是，我们将原始语料细分为 N 个子集称为折叠（folds）。
对于每一个这些的折叠，我们使用除这个折叠中的数据外其他所有数据训练模型，然后在这
个折叠上测试模型。即使个别的折叠可能是太小了而不能在其上给出准确的评价分数，综合
评估得分是基于大量的数据，因此是相当可靠的。
第二，同样重要的，采用交叉验证的优势是，它可以让我们研究不同的训练集上性能变
化有多大。如果我们从所有 N 个训练集得到非常相似的分数，然后我们可以相当有信心，
得分是准确的。另一方面，如果 N 个训练集上分数很大不同，那么，我们应该对评估得分
的准确性持怀疑态度
"""
