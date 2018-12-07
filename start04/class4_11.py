# -*- coding: utf-8 -*-
import nltk
import networkx as nx
import matplotlib

from nltk.corpus import wordnet as wn


def traverse(graph, start, node):
    graph.depth[node.name] = node.shortest_path_distance(start)
    for child in node.hyponyms():
        graph.add_edge(node.name, child.name)   # 遍历 WordNet 上位词层次为图添加边
        traverse(graph, start, child)  # 遍历是递归


def hyponym_graph(start):
    G = nx.Graph()  # 始化一个空的图
    G.depth = {}
    traverse(G, start, start)
    return G


def graph_draw(graph):
    nx.draw_networkx(graph, node_size=[16 * graph.degree(n) for n in graph],
                     node_color=[graph.depth[n] for n in graph],
                     with_labels=False)
    matplotlib.pyplot.show()

# dog = wn.synset('dog.n.01')
# graph = hyponym_graph(dog)
# graph_draw(graph)


from numpy import linalg, array

a = array([[4, 0], [3, -5]])
u, s, vt = linalg.svd(a)
print u, s, vt
