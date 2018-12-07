# -*- coding: utf-8 -*-
import nltk

# 如何确定一个词的分类
"""
形态学线索
    ness 是一个后缀，与形容词结合产生一个名词，如 happy→happiness，ill→illness。因此，如果我们遇到的一个
    以-ness 结尾的词，很可能是一个名词。同样的，-ment 是与一些动词结合产生一个名词的后缀，如 govern→government 
    和 establish→establishment一个动词的现在分词以-ing 结尾，表示正在进行的还没有结束的行动（如：falling，eating）
    的意思。-ing 后缀也出现在从动词派生的名词中，如：the falling of the leaves（这被称为动名词）。
句法线索
    另一个信息来源是一个词可能出现的典型的上下文语境。例如：假设我们已经确定了名词类。那么我们可以说，
    英语形容词的句法标准是它可以立即出现在一个名词前，或紧跟在词 be 或 very 后。根据这些测试，near 应该被归类为形容词：
    (2) a. the near window
        b. The end is (very) near

"""