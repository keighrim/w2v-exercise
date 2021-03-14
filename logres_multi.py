# ! /usr/bin/env python3

import os
import sys
from math import ceil
from typing import Dict
import logres

import numpy as np


def sigma(z):
    # scale down z values - it will help avoiding overflow for exp
    # this trick does NOT work with logistic sigmoid
    if len(z.shape) == 1:
        z = z - np.max(z)
        return np.exp(z) / np.sum(np.exp(z))
    else:
        z = z - np.max(z, axis=1)[:, None]
        return np.exp(z) / np.sum(np.exp(z), axis=1)[:, None]
    # return np.exp(z) / np.sum(np.exp(z), axis=1)[:, None]


def int_idx(things):
    return {thing: i for i, thing in enumerate(set(things))}


def onehot_idx(integer_encodings, dim=None):
    if dim is None:
        dim = np.max(integer_encodings)
    if isinstance(integer_encodings, int):
        integer_encodings = [integer_encodings]
    onehot = np.zeros((len(integer_encodings), dim))
    for row, integer in enumerate(integer_encodings):
        onehot[row, integer] = 1
    return onehot


class LogisticRegressionMulti(logres.LogisticRegression):
    class_dict: Dict
    feature_dict: Dict

    def train(self, train_set, batch_size=3, n_epochs=1, eta=0.1):
        data_mat = self.load_data(train_set)
        n_minibatches = ceil(len(data_mat) / batch_size)
        Ws_dim = (len(self.feature_dict) + 1, len(self.class_dict))
        # Ws = np.random.sample(Ws_dim)
        # Ws = np.ones(Ws_dim)
        Ws = np.zeros(Ws_dim)
        print('# features: ', len(self.feature_dict))
        for epoch in range(n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
            np.random.shuffle(data_mat)
            L = 0
            for i in range(n_minibatches):
                cur_batch = data_mat[i * batch_size: (i + 1) * batch_size]
                Xs, ys = np.split(cur_batch, [-1], axis=1)
                Ys = onehot_idx(ys.astype('int64'), dim=len(self.class_dict))
                # compute y_hat = Hs
                Zs = np.matmul(Xs, Ws)
                Hs = sigma(Zs)  #
                # update loss
                L += np.sum(np.sum(-Ys * np.log(Hs), axis=1))
                # compute gradient
                # we want dLdW, and by the chain rules, dLdW = dLdH * dHdZ * dZdW
                # note that dim of dLdW = dim of Ws, which is      (#feat      x #label)
                # from differentiation of cross entropy and softmax we know that,
                dLdH_dHdZ = Hs - Ys     # cross_entropy prime (dim: batch_size x #label), order matters
                # and
                dZdW = Xs               #                     (dim: batch_size x #feat)
                dLdW = 0
                for x in range(batch_size):
                    dLdW += np.outer(dZdW[x].T, dLdH_dHdZ[x])
                # the above for-loop is actually reduced to this dot product
                # dLdW = dZdW.T @ dLdH_dHdZ
                dLdW /= batch_size
                # update weights (and bias)
                Ws = Ws - eta * dLdW
            L /= len(data_mat)
            print("Average Train Loss: {}".format(L))
        return Ws

    def test(self, dev_set, theta):
        data_mat = self.load_data(dev_set, False)
        Xs, ys = np.split(data_mat, [-1], axis=1)
        return np.column_stack((ys, np.argmax(sigma(np.matmul(Xs, theta)), axis=1))).astype('int64')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = 'movie_reviews'
    lr = LogisticRegressionMulti()
    theta = lr.train(os.path.join(data_root, 'train'), batch_size=1, n_epochs=10, eta=0.0001)
    results = lr.test(os.path.join(data_root, 'dev'), theta)
    lr.evaluate(results)
