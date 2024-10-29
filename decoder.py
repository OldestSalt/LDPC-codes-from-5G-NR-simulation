import numpy as np
from ldpc import LDPC
from minsum import adjusted_min_sum

class Decoder:
    def __init__(self, code: LDPC):
        self.ldpc = code

    def LDPC_decode(self, iter_n: int) -> np.ndarray[float]:
        self.decoded = adjusted_min_sum(self.ldpc.H, self.llr, iter_n)
        return self.decoded
    
    def get_hard(self) -> np.ndarray[bool]:
        self.y = np.where(self.decoded < 0, True, False)
        return self.y
    
    def decode(self, llr: np.ndarray[float], iter_n: int) -> np.ndarray[bool]:
        self.llr = llr
        self.rate_dematch()
        self.LDPC_decode(iter_n)
        self.get_hard()
        self.A = self.y[:self.ldpc.k]
        return self.A
    
    def rate_dematch(self) -> np.ndarray[float]:
        self.llr = np.concatenate((np.zeros(2 * self.ldpc.Z), self.llr))
        self.ldpc.k += 2 * self.ldpc.Z
        return self.llr
