import numpy as np
from math import sqrt

class AWGNChannel:
    def __init__(self, SNR: float, R: float):
        self.snr = SNR
        self.s_squared = 1 / (2 * R * 10 ** (self.snr / 10))

    def modulate(self) -> np.ndarray[float]:
        self.modulated_word = np.where(self.initial_word, -1, 1)
        return self.modulated_word
    
    def add_awgn(self) -> np.ndarray[float]:
        self.noisy_word = np.random.normal(0, 1, self.modulated_word.shape) * sqrt(self.s_squared) + self.modulated_word
        return self.noisy_word

    def LLR(self) -> np.ndarray[float]:
        self.llr = (2 * self.noisy_word) / self.s_squared
        return self.llr

    def send_through_channel(self, word: np.ndarray[bool]) -> np.ndarray[float]:
        self.initial_word = word
        self.modulate()
        self.add_awgn()
        self.LLR()
        return self.llr