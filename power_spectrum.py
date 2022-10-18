#!/usr/bin/env python3

from datetime import datetime

import numpy as np
import numpy.fft as fft


def main():
    now = datetime.utcnow().strftime('%Y-%m-%d, %H:%M:%S')
    fname = 'iqrecord-0,0Hz-0-0'
    sr = 2400000
    k = sr // 800
    chunk_size = 1
    while chunk_size < k:
        chunk_size *= 2
    data = np.fromfile(fname, dtype=np.int8)
    data_c = data.astype(np.float32).view(np.complex64) / 127
    power = np.zeros(chunk_size)
    nchunks = len(data_c) // chunk_size
    for n in range(0, nchunks):
        idx = n * chunk_size
        f = fft.fft(data_c[idx:idx+chunk_size])
        power += np.real(np.conj(f) * f)
    power = fft.fftshift(power)
    power /= nchunks * chunk_size
    p = 10 * np.log10(power)
    np.savetxt(
        'spectrum.txt',
        p,
        fmt='%.2f',
        newline=', ',
        header='%s FREW_LO, FREQ_HI, %f.2, %d' % (now, sr/chunk_size, nchunks,),
        comments=''
    )


if __name__ == '__main__':
    main()
