import math

import logger

import numpy as np
import pandas as pd


class Neiron:
    def __init__(self, dimenson):
        self.w = np.random.randn(dimenson)

    def update(self, delta_gain):
        self.w = self.w + delta_gain

class MySelfOrginizeMap:
    def __init__(self, n1, n2, w2, n=10, b_ij=None, sigma=None, h=None):
        """
        Инициализация всех констант, и функций
        :param n1: икс по в карте
        :param n2: игрик по в карте
        :param w2: количество осей в  исходных данных по дефолту равен n1
        :param int n: предел итераций
        :param b_ij: функция соседства
        :param sigma: радиус функции соседства
        """
        self.n = n
        self.code = 0
        self.d = min(n1, n2, w2)/2 # постоянная временииспользуемая для уменьшения радиуса скорости обучения

        if b_ij is None:
            self.b_ij = self._my_b_ij
        else:
            self.b_ij = b_ij

        if sigma is None:
            self.sigma = self._sigma
        else:
            self.sigma = sigma

        if h is None:
            # simple constant
            self.h = self._h_gauss
        else:
            self.h = h

        brain = []
        for i in range(n1 * n2):
            brain.append(Neiron(w2))
        self.brain = brain

    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            for i, x in enumerate(data.values):
                if i + 1 == self.n:
                    break

                best = -1
                winner = -1
                for ind, neu in enumerate(self.brain):
                    predict_score = self.b_ij(x, neu.w)
                    if predict_score < best or best == -1:
                        winner = ind
                        best = predict_score

                # change and update all weight
                dd = self.h(self.d, i)
                for ind, neu in enumerate(self.brain):
                    if winner == ind:
                        continue
                    else:
                        delta_gain = dd * (x - neu.w)
                        neu.update(delta_gain)
                print(f'finish epoch=====================>{i}')
        elif isinstance(data, np.array([12])):
            logger.error('We havent some ralization')
        else:
            logger.error('incorrect type og input value')
        for i in self.brain:
            print(i.w)

    def _my_b_ij(self, arr1, arr2):
        if len(arr1) != len(arr2):
            raise
        ans = 0
        for i in range(len(arr1)):
            ans = (arr1[i] - arr2[i]) ** 2
        return ans

    def _sigma(self, t):
        return math.exp(-t / self.d)

    def _h_constructor(self, d, t):
        print(self.sigma(t))
        if self.sigma(t) > d:
            return self.code
        else:
            return 0

    def _h_gauss(self, d, t):
        p = -d / (2 * self.sigma(t))
        return math.e ** p

if __name__ == "__main__":
    table = np.array([
        [0, 0, 0],
        [1, 1, 2],
        [-1, -1, -1],
        [-1, 1, 0],
        [1, -1, 9],
        [0, 2, 2],
        [0, -2, -2]
    ])
    df = pd.DataFrame(table, columns=['par1', 'par2', 'p3'])
    mdl = MySelfOrginizeMap(2, 2, len(table[0]))
    mdl.fit(df)
