# -*- coding: utf-8 -*-
import math
import nltk

# 6.5 朴素贝叶斯分类器
"""
在朴素贝叶斯分类器中，每个特征都得到发言权，来确定哪个标签应该被分配到一个给
定的输入值。为一个输入值选择标签，朴素贝叶斯分类器以计算每个标签的先验概率开始，
它由在训练集上检查每个标签的频率来确定。之后，每个特征的贡献与它的先验概率组合，
得到每个标签的似然估计。似然估计最高的标签会分配给输入值。
在训练语料中，大多数文档是有关汽车的，所以分类器从接近“汽车”的标签的点上开始。但它会考虑每个特征的
影响。在这个例子中，输入文档中包含的词 dark，它是谋杀之谜的一个不太强的指标，也
包含词 football，它是体育文档的一个有力指标。每个特征都作出了贡献之后，分类器检查
哪个标签最接近，并将该标签分配给输入。
个别特征对整体决策作出自己的贡献，通过“投票反对”那些不经常出现的特征的标签。
特别是，每个标签的似然得分由于与输入值具有此特征的标签的概率相乘而减小。例如：如
果词 run 在 12%的体育文档中出现，在 10%的谋杀之谜的文档中出现，在 2％的汽车文档中
出现，那么体育标签的似然得分将被乘以 0.12，谋杀之谜标签将被乘以 0.1，汽车标签将被
乘以 0.02。整体效果是：略高于体育标签的得分的谋杀之谜标签的得分会减少，而汽车标签
相对于其他两个标签会显著减少
"""
# 潜在概率模型

"""
理解朴素贝叶斯分类器的另一种方式是它为输入选择最有可能的标签，基于下面的假设：
每个输入值是通过首先为那个输入值选择一个类标签，然后产生每个特征的方式产生的，
每个特征与其他特征完全独立。当然，这种假设是不现实的，特征往往高度依赖彼此。我们
将在本节结尾回过来讨论这个假设的一些后果。这简化的假设，称为朴素贝叶斯假设（或独
立性假设），使得它更容易组合不同特征的贡献，因为我们不必担心它们相互影响。基于这个假设，
我们可以计算表达式 P(label|features)，给定一个特别的特征集一个输入具有特定标签的概率。
要为一个新的输入选择标签，我们可以简单地选择使 P(l|features)最大的标签 l。
一开始，我们注意到 P(label|features)等于具有特定标签和特定特征集的输入的概率除以
具有特定特征集的输入的概率：
(2) P(label|features) = P(features, label)/P(features)
接下来，我们注意到 P(features)对每个标签选择都相同。因此，如果我们只是对寻找最
有可能的标签感兴趣，只需计算 P(features,label)，我们称之为该标签的似然
如果我们想生成每个标签的概率估计，而不是只选择最有可能的标签，那么计算 P(features)的最简单的方法是简单的计算
 P(features, label)在所有标签上的总和：
(3) P(features) = Σlabel ∈ labels P(features, label)
标签的似然可以展开为标签的概率乘以给定标签的特征的概率：
(4) P(features, label) = P(label) × P(features|label)
此外，因为特征都是独立的（给定标签），我们可以分离每个独立特征的概率：
(5) P(features, label) = P(label) × ∏f ∈ featuresP(f|label)
这正是我们前面讨论的用于计算标签可能性的方程式：P(label)是一个给定标签的先验
概率，每个 P(f|label)是一个单独的特征对标签可能性的贡献。
"""
# 零计数和平滑
"""
最简单的方法计算 P(f|label)，特征 f 对标签 label 的标签可能性的贡献，是取得具有给
定特征和给定标签的训练实例的百分比：
(6) P(f|label) = count(f, label)/count(label)
然而，当训练集中有特征从来没有和给定标签一起出现时，这种简单的方法会产生一个
问题。在这种情况下，我们的 P(f|label)计算值将是 0，这将导致给定标签的标签可能性为 0。
从而，输入将永远不会被分配给这个标签，不管其他特征有多么适合这个标签。
这里的基本问题与我们计算 P(f|label)有关，对于给定标签输入将具有一个特征的概率。
特别的，仅仅因为我们在训练集中没有看到特征/标签组合出现，并不意味着该组合不会出
现。例如：我们可能不会看到任何谋杀之迷文档中包含词 football，但我们不希望作出结论
认为在这些文档中存在是完全不可能。
虽然 count(f,label)/count(label)当 count(f,label)相对高时是 P(f|label)的好的估计，当 coun
t(f)变小时这个估计变得不那么可靠。因此，建立朴素贝叶斯模型时，我们通常采用更复杂
的技术，被称为平滑技术，用于计算 P(f|label)，给定标签的特征的概率。例如：给定标签的
一个特征的概率的期望似然估计基本上给每个 count(f,label)值加 0.5，Heldout估计使用一个
heldout 语料库计算特征频率与特征概率之间的关系。nltk.probability 模块提供了多种平
滑技术的支持
"""
# 非二元特征
"""
我们这里假设每个特征是二元的，即每个输入要么有这个特征要么没有。标签值特征（例
如：颜色特征，可能有红色、绿色、蓝色、白色或橙色）可通过用二元特征，如“颜色是红
色”，替换它们，将它们转换为二元特征。数字特征可以通过装箱转换为二元特征，装箱是
用特征，如“4<X <6”，替换它们。
另一种方法是使用回归方法模拟数字特征的概率。例如：如果我们假设特征 height 具有
贝尔曲线分布，那么我们可以通过找到每个标签的输入的 height 的均值和方差来估算 P(hei
ght|label)。在这种情况下，P(f=v|label)将不会是一个固定值，会依赖 v 的值变化
"""

# 独立的朴素

"""
朴素贝叶斯分类器被称为“naive（天真、朴素）”的原因是它不切实际地假设所有特征
相互独立（给定标签）。特别的，几乎所有现实世界的问题含有的特征都不同程度的彼此依
赖。如果我们要避免任何依赖其他特征的特征，那将很难构建良好的功能集，提供所需的信
息给机器学习算法。
如果我们忽略了独立性假设，使用特征不独立的朴素贝叶斯分类器会发生什么？产生的
一个问题是分类器“双重计数”高度相关的特征的影响，将分类器推向更接近给定的标签而
不是合理的标签。
来看看这种情况怎么出现的，思考一个包含两个相同的特征 f1 和 f2 的名字性别分类器。
换句话说，f2 是 f1 的精确副本，不包含任何新的信息。当分类器考虑输入，在决定选择哪
一个标签时，它会同时包含 f1 和 f2 的贡献。因此，这两个特征的信息内容将被赋予比它们
应得的更多的比重。
当然，我们通常不会建立包含两个相同的特征的朴素贝叶斯分类器。不过，我们会建立
包含相互依赖特征的分类器。例如：特征 ends-with(a)和 ends-with(vowel)是彼此依赖，
因为如果一个输入值有第一个特征，那么它也必有第二个特征。对于这些功能，重复的信息
可能会被训练集赋予比合理的更多的比重。
"""
# 双重计数的原因
"""
双重计数问题的原因是在训练过程中特征的贡献被分开计算，但当使用分类器为新输入
选择标签时，这些特征的贡献被组合。因此，一个解决方案是考虑在训练中特征的贡献之间
可能的相互作用。然后，我们就可以使用这些相互作用调整独立特征所作出的贡献。
为了使这个更精确，我们可以重写计算标签的可能性的方程，分离出每个功能（或标签）
所作出的贡献
(7) P(features, label) = w[label] × ∏f ∈ features w[f, label]
在这里，w[label]是一个给定标签的“初始分数”，w[f, label]是给定特征对一个标签的
可能性所作的贡献。我们称这些值 w[label]和 w[f, label]为模型的参数或权重。使用朴素贝
叶斯算法，我们单独设置这些参数：
(8) w[label] = P(label)
(9) w[f, label] = P(f|label)
"""