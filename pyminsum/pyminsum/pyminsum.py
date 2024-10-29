import numpy as np
from awgn import AWGNChannel
from encoder import Encoder
from minsum import adjusted_min_sum




def write_matrix(pm):
    fi = open('test.txt', 'w')
    for row in pm:
        for el in row:
            fi.write(str(el) + ' ')
        fi.write('\n')
    fi.close()
    return
code = Encoder(3 / 4, 20)
word = np.random.randint(0, 2, 20).astype(bool)
code.encode(word)
code.generate_parity_matrix(2)
Z_c = 2
code.LDPC_encode(Z_c)
# word = code.D
word = code.D_full
# print(word.astype(int))
channel = AWGNChannel(2, 4, 1 / 4)
llr = channel.send_through_channel(word)
# print(code.parity_matrix.astype(int).shape)
# print(llr.shape, code.parity_matrix.astype(int).shape)

decoded = adjusted_min_sum(code.parity_matrix.astype(int), llr, 30)
print(llr)
input(decoded)