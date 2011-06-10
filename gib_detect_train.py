#!/usr/bin/python

import math
import pickle

accepted_chars = 'abcdefghijklmnopqrstuvwxyz '

pos = dict([(char, idx) for idx, char in enumerate(accepted_chars)])

def normalize(line):
    return [c.lower() for c in line if c.lower() in accepted_chars]

def ngram(n, l):
    filtered = normalize(l)
    for start in range(0, len(filtered) - n + 1):
        yield ''.join(filtered[start:start + n])

def train():
    k = len(accepted_chars)
    counts = [[10 for i in xrange(k)] for i in xrange(k)]

    for line in open('big.txt'):
        for a, b in ngram(2, line):
            # print a, b, counts[pos[a]][pos[b]]
            counts[pos[a]][pos[b]] += 1

    for i, row in enumerate(counts):
        s = float(sum(row))
        for j in xrange(len(row)):
            row[j] = math.log(row[j] / s)
            # print i, j, accepted_chars[i], accepted_chars[j], row[j]

    good_ents = [char_ent(l, counts) for l in open('good.txt')]
    bad_ents = [char_ent(l, counts) for l in open('bad.txt')]

    good_ents = sorted(good_ents)
    bad_ents = sorted(bad_ents)

    assert min(good_ents) > max(bad_ents)

    thresh = (min(good_ents) + max(bad_ents)) / 2
    pickle.dump({'mat': counts, 'thresh': thresh}, open('gib_model.pki', 'wb'))

def char_ent(l, count_mat):
    ret = 1.0
    length = len(normalize(l)) or 1
    for a, b in ngram(2, l):
        # print a, b, counts[pos[a]][pos[b]], ret
        ret += count_mat[pos[a]][pos[b]]
    return math.exp(ret / length)

if __name__ == '__main__':
    train()



    
    
