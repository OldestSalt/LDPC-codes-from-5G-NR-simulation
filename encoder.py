import numpy as np
import pandas as pd
from ldpc import LDPC
from math import ceil

Z_index_sets = [
    [2, 4, 8, 16, 32, 64, 128, 256],
    [3, 6, 12, 24, 48, 96, 192, 384],
    [5, 10, 20, 40, 80, 160, 320],
    [7, 14, 28, 56, 112, 224],
    [9, 18, 36, 72, 144, 288],
    [11, 22, 44, 88, 176, 352],
    [13, 26, 52, 104, 208],
    [15, 30, 60, 120, 240]
]


class Encoder:
    def __init__(self, code: LDPC):
        self.ldpc = code

    def encode(self, input_word: np.ndarray[bool]) -> np.ndarray[bool]:
        self.A = input_word
        self.LDPC_encode()
        self.rate_match()
        return self.D
    
    def CRC(self):
        pass

    def segmentation(self):
        pass

    def LDPC_encode(self):
        # C = self.A.copy()
        self.D = np.zeros(self.ldpc.H.shape[1] - 2 * self.ldpc.Z, dtype=bool)
        # self.D_full = np.zeros(self.ldpc.H.shape[1], dtype=bool)

        # C[C == None] = False
        X = self.solve_parity_bits()
        self.D = X[2 * self.ldpc.Z:]
        # self.D_full = X
    
    def solve_parity_bits(self) -> np.ndarray[bool]:
        N = self.ldpc.H.shape[1]
        K = len(self.A)
        A = self.ldpc.H.copy()
        Z_c = self.ldpc.Z

        X = np.zeros(N, dtype=bool)
        # Get rid of first K variables
        X[:K] = self.A.copy()
        B = np.zeros(A.shape[0], dtype=bool)
        for i in range(K):
            if X[i]:
                B ^= A[:, i]
        # Gauss method is needed to solve equation system from submatrix B
        pm_sub = A[:4 * Z_c, K:K + 4 * Z_c]
        B_sub = B[:4 * Z_c]

        # Forward
        for i in range(4 * Z_c):
            if not pm_sub[i, i]:
                for j in range(i + 1, 4 * Z_c):
                    if pm_sub[j, i]:
                        pm_sub[[j, i]] = pm_sub[[i, j]]
                        B_sub[[j, i]] = B_sub[[i, j]]
                        break
            for k in range(i + 1, 4 * Z_c):
                if pm_sub[k, i]:
                    pm_sub[k] ^= pm_sub[i]
                    B_sub[k] ^= B_sub[i]
        
        # Backward
        for i in range(4 * Z_c - 1, -1, -1):
            for j in range(i):
                if pm_sub[j, i]:
                    pm_sub[j] ^= pm_sub[i]
                    B_sub[j] ^= B_sub[i]

        X[K:K + 4 * Z_c] = B_sub
        B[:4 * Z_c] = B_sub
        A[:4 * Z_c, K:K + 4 * Z_c] = pm_sub

        for i in range(K, K + 4 * Z_c):
            if X[i]:
                B ^= A[:, i]
        
        # Only diagonal matrix left, so the calculation of remaining variables is trivial
        X[K + 4 * Z_c:] = B[4 * Z_c:]
        return X

    def rate_match(self) -> np.ndarray[bool]:
        target_n = ceil(self.ldpc.k / self.ldpc.r)
        self.D = self.D[:target_n]
        self.ldpc.H = self.ldpc.H[:target_n - (self.ldpc.H.shape[1] - self.ldpc.H.shape[0]) + 2 * self.ldpc.Z, :target_n + 2 * self.ldpc.Z]
        self.ldpc.k = self.ldpc.H.shape[1] - self.ldpc.H.shape[0] - 2 * self.ldpc.Z
        return self.D