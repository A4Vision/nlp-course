from collections import defaultdict
from itertools import chain

from PCFG import PCFG
import math

def load_sents_to_parse(filename):
    sents = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                sents.append(line)
    return sents

def cky(pcfg, sent):
    ### YOUR CODE HERE
    split_sent = sent.split()
    n = len(split_sent)
    pi = defaultdict(float)
    bp = {}
    rules = pcfg._rules
    N = set(rules.keys() + list(chain.from_iterable(map(lambda x: x[0][0] if len(x) > 0 else [], rules.values()))))
    for i in range(1, n + 1):
        set_flag = False
        curr_word = split_sent[i - 1]
        for X in N:
            derivations = rules[X]
            for derivation in derivations:
                if [curr_word] == derivation[0]:
                    pi[(i, i, X)] = derivation[1]
                    bp[(i, i, X)] = Node(X, None, None, curr_word)
                    set_flag = True
        if not set_flag:
            pi[(i, i, curr_word)] = 0.0000000001
            bp[(i, i, curr_word)] = Node(None, None, None, curr_word)
    for i in reversed(range(1, n + 1)):
        for l in range(1, n - i + 1):
            j = i + l
            for X in N:
                derivations = rules[X]
                max_value = 0
                max_input = None
                for derivation in derivations:
                    for s in range(i, j):
                        if len(derivation[0]) == 2:
                            value = derivation[1] * pi[(i, s, derivation[0][0])] * pi[(s + 1, j, derivation[0][1])]
                            max_value = max(value, max_value)
                            if value == max_value and value > 0:
                                max_input = Node(X, bp[(i, s, derivation[0][0])],
                                                 bp[(s + 1, j, derivation[0][1])], None)
                pi[(i, j, X)] = max_value
                if max_input:
                    bp[(i, j, X)] = max_input
    for key in bp:
        node = bp[key]
        if node.root == 'ROOT' and key[1] - key[0] == len(split_sent) - 1:
            return get_tree(node)
    ### END YOUR CODE
    return "FAILED TO PARSE!"


def get_tree(root):
    """
    getParseTree() takes a root and constructs the tree in the form of a
    string. Right and left subtrees are indented equally, providing for
    a nice display.
    @params: root node and an indent factor (int).
    @return: tree, starting at the root provided, in the form of a string.
    """
    if root.terminal and root.root:
        return '(' + root.root + ' ' + root.terminal + ')'
    if root.terminal:
        return root.terminal

    # Calculates the new indent factors that we need to pass forward.
    left = get_tree(root.left)
    right = get_tree(root.right)
    return '(' + root.root + ' ' + left + ' ' + right + ')'


class Node:
    def __init__(self, root, left, right, terminal):
        self.root = root
        self.left = left
        self.right = right
        self.terminal = terminal


if __name__ == '__main__':
    import sys
    pcfg = PCFG.from_file_assert_cnf(sys.argv[1])
    sents_to_parse = load_sents_to_parse(sys.argv[2])
    for sent in sents_to_parse:
        print cky(pcfg, sent)
