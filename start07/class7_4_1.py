#  -*- coding: utf-8  -*-
import nltk

# 7.4 语言结构中的递归 语言结构中的递归 语言结构中的递归 语言结构中的递归
# 用级联分块器构建嵌套结构
"""
我们的块结构一直是相对平的。已标注标识符组成的树在如 NP 这样的块节点下任意组合。然而，
只需创建一个包含递归规则的多级的块语法，就可以建立任意深度的块结构。例 7-6 是名词短语、介词短语、动词短语和句子的模式。
这是一个四级块语法器，可以用来创建深度最多为 4 的结构
"""
# 例 7-6. 一个分块器，处理 NP，PP，VP 和 S。
grammar = r"""
        NP: {<DT|JJ|NN.*>+} # Chunk sequences of DT, JJ, NN
        PP: {<IN><NP>} # Chunk prepositions followed by NP
        VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
        CLAUSE: {<NP><VP>} # Chunk NP, VP
"""
cp = nltk.RegexpParser(grammar)
sentence = [("Mary", "NN"), ("saw", "VBD"), ("the", "DT"), ("cat", "NN"),
            ("sit", "VB"), ("on", "IN"), ("the", "DT"), ("mat", "NN")]
print cp.parse(sentence)
sentence = [("John", "NNP"), ("thinks", "VBZ"), ("Mary", "NN"), ("saw", "VBD"), ("the", "DT"),
            ("cat", "NN"), ("sit", "VB"), ("on", "IN"), ("the", "DT"), ("mat", "NN")]
cp = nltk.RegexpParser(grammar, loop=2)  # 两次迭代
print cp.parse(sentence)

# 树
# 树是一组连接的加标签节点，从一个特殊的根节点沿一条唯一的路径到达每个节点。下面是一棵树的例子
# （注意：它们标准的画法是颠倒的）：
"""
我们用“家庭”来比喻树中节点的关系：例如：S 是 VP 的父母；反之 VP 是 S 的一个孩
子。此外，由于 NP 和 VP 同为 S 的两个孩子，它们也是兄弟。为方便起见，也有特定树的
文本格式：
(S
    (NP Alice)
    (VP
        (V chased)
        (NP
            (Det the)
            (N rabbit))))
在 NLTK 中，我们创建了一棵树，通过一个节点添加标签和一个孩子链表
"""
tree1 = nltk.Tree('NP', ['Alice'])
print tree1
tree2 = nltk.Tree('NP', ['the', 'rabbit'])
print tree2
tree3 = nltk.Tree('VP', ['chased', tree2])
tree4 = nltk.Tree('S', [tree1, tree3])
print tree4
print tree4[1]
print tree4[1].label()
print tree4.leaves()
print tree4.draw()


# 树遍历

# 使用递归函数来遍历树是标准的做法。例 7-7 中的列表进行了演示。
# 例 7-7. 递归函数遍历树。
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

t = nltk.Tree('S', ['NP', ['Alice'], ['VP', ['chased', 'NP', ['the rabbit']]]])
# print t
traverse(t)
