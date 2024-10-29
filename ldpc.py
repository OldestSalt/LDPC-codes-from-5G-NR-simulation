import numpy as np
import pandas as pd
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

class LDPC:
    def __init__(self, r: int, k: int):
        self.MIN_K = 20
        self.MAX_K = 8448
        if k <= self.MAX_K and k >= self.MIN_K:
            self.k = k
        else:
            raise ValueError('Information word length must be less than 3840')
        
        self.MAX_R = 948 / 1024
        self.MIN_R = 1 / 3
        if r <= self.MAX_R and r >= self.MIN_R:
            self.r = r
        else:
            raise ValueError('LDPC code\'s target rate cannot be more than 948/1024 and less than 1/3')
    
    def generate_parity_matrix(self) -> np.ndarray[bool]:
        # Base graph choice depends on information word length, until segmentation and CRC are applied
        if self.k <= 292 or (self.k <= 3824 and self.r <= 0.67) or self.r <= 0.25:
            bg_number = '2'
            base_k = 10
        else:
            bg_number = '1'
            base_k = 22

        BG = pd.read_csv(f'bg{bg_number}.csv', header=None).ffill()
        BG = BG.set_index([0, 1])
        BG.columns = range(8)
        
        Z_near = self.k / base_k
        i_LS, Z_c = self.get_i_LS(Z_near)
        if i_LS >= 0:
            bg = BG[i_LS].unstack().values
            self.Z = Z_c
            self.k -= 2 * Z_c
        else:
            raise ValueError('There is no fitting Z value')
        
        pm = np.zeros((Z_c * bg.shape[0], Z_c * bg.shape[1]), dtype=bool)
        for i in range(bg.shape[0]):
            for j in range(bg.shape[1]):
                if np.isnan(bg[i, j]):
                    block = np.zeros((Z_c, Z_c), dtype=bool)
                else:
                    block = np.roll(np.eye(Z_c, dtype=bool), int(bg[i, j]) % Z_c, axis=1)
                pm[i * Z_c:(i + 1) * Z_c, j * Z_c:(j + 1) * Z_c] = block
        
        self.H = pm
        return pm

    @staticmethod
    def get_i_LS(Z_near: int) -> tuple[int, int]:
        min_index = -1
        min_delta = 500
        Z_c = -1
        for index, Z_set in enumerate(Z_index_sets):
            deltas = np.array(Z_set) - Z_near
            deltas = deltas[deltas >= 0]
            curr_min = deltas.min()

            if curr_min < min_delta:
                min_delta = curr_min
                min_index = index
                Z_c = curr_min + Z_near

        return (min_index, int(Z_c))