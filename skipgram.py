# ! /usr/bin/env python3

import os
import sys
from typing import Dict, Any, Union
from logres_multi import sigma as softmax, onehot_idx, int_idx

import numpy as np


def sliding_window(dataset, win_size=1):
    for document in dataset:
        for i, word in enumerate(document):
            s = (max(i - win_size, 0))
            e = (min(i + win_size + 1, len(document)))
            yield [word] + document[s:i] + document[i+1:e]


def cosine_dist(a, b):
    return a@b / np.linalg.norm(a) * np.linalg.norm(b)


class Skipgram(object):
    embedding: np.ndarray
    word_idx: Dict
    inv_word_idx: Dict

    def load_data(self, data_set):
        tokens = set()
        for root, dirs, files in os.walk(data_set):
            for name in [f for f in files if f.endswith('.txt')]:
                with open(os.path.join(root, name)) as doc_f:
                    tokens.update(token for token in doc_f.read().split())
        self.word_idx = int_idx(tokens)
        self.inv_word_idx = {v: k for k, v in self.word_idx.items()}

        documents = []
        for root, dirs, files in os.walk(data_set):
            for name in [f for f in files if f.endswith('.txt')]:
                with open(os.path.join(root, name)) as doc_f:
                    document = [self.word_idx[wrd] for wrd in doc_f.read().split()]
                    documents.append(document)
        return documents

    def train(self, data_root, embdding_size=100, n_epochs=1, eta=0.001):
        data = self.load_data(data_root)
        vocab_size = len(self.word_idx)
        N = embdding_size
        V = vocab_size
        report_every = 200

        # weight matrix for target (center) words
        Wt_dim = (V, N)
        # weight matrix for context word
        Wc_dim = (N, V)
        Wc = np.random.sample(Wc_dim) / 100
        Wt = np.random.sample(Wt_dim) / 100
        windows = list(sliding_window(data, win_size=1))
        for epoch in range(n_epochs):
            np.random.shuffle(windows)
            print('training examples:', len(windows))

            L = 0
            for i, window in enumerate(windows, 1):
                if i % report_every == 0:
                    print(f'{report_every}-window average loss: ', L / report_every)
                    L = 0
                X = onehot_idx(window[0], V)
                Ys = onehot_idx(window[1:], V)

                # feed forward
                F = X@Wt      # as in features
                Z = F@Wc
                H = softmax(Z)

                L += -np.sum(Z.T[Ys.nonzero()[1]]) + len(Ys) * np.log(np.sum(np.exp(Z)))

                # first we want dLdWc at the last layer. This is identical to multinomial logistic regression
                # except for that Ys has many columns. So we sum all differences along columns.
                dLdH_dHdZ = np.sum(H - Ys, axis=0)      # V
                dZdWc = F
                dLdWc = np.outer(dZdWc.T, dLdH_dHdZ)     # NxV
                Wc -= eta * dLdWc

                # now we want dLdWt, and by the chain rule, dLdWt = dLdZ * dZdF * dFdWt
                dLdZ = dLdH_dHdZ                    # "delta" back-propagated from last layer
                dZdF = Wc
                dLdF = dZdF @ dLdZ                  # V @ NxV
                dFdWt = X
                dLdWt = np.outer(dLdF, dFdWt).T     # VxN
                Wt -= eta * dLdWt

        self.embedding = Wt
        return Wt

    def get_vector(self, word: str):
        return self.embedding[self.word_idx[word]]

if __name__ == '__main__':
    data_root = "gutenberg-austen"
    w2v = Skipgram()
    w2v.train(data_root)
