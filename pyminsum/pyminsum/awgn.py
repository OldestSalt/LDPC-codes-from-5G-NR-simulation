import numpy as np
from math import sqrt

class AWGNChannel:
    def __init__(self, M: int, SNR: float, R: float):
        self.snr = SNR
        self.M = M
        self.s_squared = 1 / (2 * R * 10 ** (self.snr / 10))

    def modulate(self, word: np.ndarray[bool]) -> np.ndarray[float]:
        if (self.M > 2):
            k = int(np.log2(self.M))
            gray_code = self.get_gray_code(k).dot(1 << np.arange(k)[::-1])
            word = np.reshape(word, (-1, k)).astype(int).dot(1 << np.arange(k)[::-1]) # Не забыть учесть случай если длина блока не делится на блоки при модуляции
            phase = 2 * np.pi / self.M

            angles = np.array([np.where(gray_code == ch)[0][0] for ch in word]) * phase
            complex_coords = np.cos(angles)[:, np.newaxis]
            complex_coords = np.hstack((complex_coords, np.sin(angles)[:, np.newaxis]))
            return complex_coords
        else:
            return np.array([-1 if ch else 1 for ch in word], dtype=float)
    
    def add_awgn(self, word: np.ndarray[float]) -> np.ndarray[float]:
        if self.M > 2:
           s = self.s_squared / 2
        else:
            s = self.s_squared
        return np.random.normal(0, 1, word.shape) * sqrt(s) + word
    
    def demodulate(self, word): # Не нужна для бинарной фазовой модуляции, пусть будет пока заготовка
        pass

    def LLR(self, word: np.ndarray[float]) -> np.ndarray[float]:
        return (2 * word) / self.s_squared

    def send_through_channel(self, word: np.ndarray[bool]) -> np.ndarray[float]: # Временно добавлю сюда LLR, уберу, если не надо
        # Пока предполагаем, что модуляция только бинарная
        word = self.modulate(word)
        word = self.add_awgn(word)
        # word = self.LLR(word)
        return word


    @staticmethod
    def get_gray_code(n: int) -> np.ndarray[int]:
        if n > 1:
            code = AWGNChannel.get_gray_code(n - 1)
            rev_code = code[::-1]
            rev_code = np.hstack((np.ones((rev_code.shape[0], 1), dtype=int), rev_code))
            code = np.hstack((np.zeros((code.shape[0], 1), dtype=int), code))
            return np.concatenate((code, rev_code))
        else:
            return np.array([[0], [1]])