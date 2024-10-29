import numpy as np
import pandas as pd


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
    def __init__(self, r: int, k: float):
        self.k = k
        self.R = r
    
    def generate_parity_matrix(self, Z_c: int) -> np.ndarray[bool]:
        if len(self.A) <= 292 or (len(self.A) <= 3824 and self.R <= 0.67) or self.R <= 0.25:
            BG = pd.read_csv('bg2.csv', header=None).ffill()
        else:
            BG = pd.read_csv('bg1.csv', header=None).ffill()
        
        BG = BG.set_index([0, 1])
        BG.columns = range(8)
        
        i_LS = self.get_i_LS(Z_c)
        if i_LS >= 0:
            bg = BG[i_LS].unstack().values
        else:
            raise ValueError('There is no such Z value')
        
        pm = np.zeros((Z_c * bg.shape[0], Z_c * bg.shape[1]), dtype=bool)
        for i in range(bg.shape[0]):
            for j in range(bg.shape[1]):
                if np.isnan(bg[i, j]):
                    block = np.zeros((Z_c, Z_c), dtype=bool)
                else:
                    block = np.roll(np.eye(Z_c, dtype=bool), int(bg[i, j]) % Z_c, axis=1)
                pm[i * Z_c:(i + 1) * Z_c, j * Z_c:(j + 1) * Z_c] = block
        
        self.parity_matrix = pm
        return pm

    @staticmethod
    def get_i_LS(Z: int) -> int:
        for index, Zs in enumerate(Z_index_sets):
            if Z in Zs:
                return index
        return -1

    def encode(self, input_block: np.ndarray[bool]):
        self.A = input_block

        return 
    
    def CRC(self):
        pass

    def segmentation(self):
        pass

    def LDPC_encode(self, Z_c: int) -> np.ndarray[bool]:
        C = self.A.copy()
        self.D = np.zeros(self.parity_matrix.shape[1] - 2 * Z_c, dtype=bool)
        self.D_full = np.zeros(self.parity_matrix.shape[1], dtype=bool)

        # self.D[:len(C) - 2 * Z_c] = C[2 * Z_c:]
        C[C == None] = False
        X = self.solve_parity_bits(Z_c, C)
        # self.D[len(C) - 2 * Z_c:] = X[len(C):]
        self.D = X[2 * Z_c:]
        self.D_full = X
    
    def solve_parity_bits(self, Z_c: int, C: np.ndarray[bool]) -> np.ndarray[bool]:
        N = self.parity_matrix.shape[1]
        K = len(C)
        A = self.parity_matrix.copy()

        X = np.zeros(N, dtype=bool)
        # Избавляемся от первых K символов
        X[:K] = C
        B = np.zeros(A.shape[0], dtype=bool)
        for i in range(K):
            if X[i]:
                B ^= A[:, i]
        # Метод Гаусса для решение системы подматрицы B размером 4 * Z_c
        pm_sub = A[:4 * Z_c, K:K + 4 * Z_c]
        B_sub = B[:4 * Z_c]

        # Прямой ход
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
        
        # Обратный ход
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
        
        # Осталась только диагональная матрица, приравниваем оставшие символы
        X[K + 4 * Z_c:] = B[4 * Z_c:]
        return X