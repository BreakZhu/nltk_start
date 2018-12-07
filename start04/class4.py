# -*- coding: utf-8 -*-

import nltk


def search1(substring, words):
    result = []
    for word in words:
        if substring in word:
            result.append(word)
    return result


def search2(substring, words):
    for word in words:
        if substring in word:
            yield word


# print "search1:"
# for item in search1('zz', nltk.corpus.brown.words()):
#     print item
# print "search2:"
# for item in search2('zz', nltk.corpus.brown.words()):
#     print item


def permutations(seq):
    if len(seq) <= 1:
        yield seq
    else:
        for perm in permutations(seq[1:]):  # 递归
            for i in range(len(perm) + 1):
                yield perm[:i] + seq[0:1] + perm[i:]

# print list(permutations(['police', 'fish', 'buffalo']))

import re


def raw(file):
    """
    :param file:
    :return:
    """
    contents = open(file, "rb").read()
    return str(contents)


def snippet(doc, term):  # buggy
    text = ' ' * 30 + raw(doc) + ' ' * 30
    pos = text.index(term)
    return text[pos - 30:pos + 30]


print "Building Index..."
files = nltk.corpus.movie_reviews.abspaths()
idx = nltk.Index((w, f) for f in files for w in raw(f).split())
query = ''
while query != "quit":
    query = raw_input("query> ")
    if query in idx:
        for doc in idx[query]:
            print snippet(doc, query)
    else:
        print "Not found"
