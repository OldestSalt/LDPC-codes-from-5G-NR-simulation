import numpy as np
from awgn import AWGNChannel
from encoder import Encoder
from decoder import Decoder
from ldpc import LDPC
import pandas as pd
from math import log10

def simulate_with_params(k: int, r: float, max_iter: int) -> pd.DataFrame:
    ldpc = LDPC(r, k)
    encoder = Encoder(ldpc)
    decoder = Decoder(ldpc)
    count = 0
    results = []

    print(f'k = {k}, r = {r:.3f}')
    SNR = 0
    last_FER = 1
    while SNR < 10 and last_FER >= 1e-3:
        channel = AWGNChannel(SNR, r)
        count = 0
        n = 0

        while count < 100 and n < 10000:
            print(f'{n} test')
            n += 1
            word = np.random.randint(0, 2, k).astype(bool)
            ldpc.generate_parity_matrix()
            encoder.encode(word)
            llr = channel.send_through_channel(encoder.D)
            result = decoder.decode(llr, max_iter)
            if (not np.all(word == result)):
                count += 1
                print('FAIL')
            else:
                print('SUCCESS')

        FER = count / n

        if (last_FER - FER < 1e-4):
            step = 0.1
        else:
            step = 0.1 * (-np.log10(last_FER - FER) + 1)

        last_FER = FER
        results.append((k, r, SNR, FER, n))
        print(f'SNR = {SNR:.1f}, FER = {FER:.4f}, {n} tests')
        SNR += step
    
    return pd.DataFrame(results, columns=['k', 'r', 'SNR', 'FER', 'n_tests'])

def run_simulation(params: list[tuple[int, float]], max_iter: int):
    for k, r in params:
        res = simulate_with_params(k, r, max_iter)
        res.to_csv(f'tests_results/len{k}rate{r:.2f}.csv')
        print(f'len{k}rate{r:.2f}.csv is saved!')