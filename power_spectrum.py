#!/usr/bin/env python3

import subprocess
from datetime import datetime

import numpy as np
import numpy.fft as fft


def get_channel_info():
    output = subprocess.check_output(
        './metadump -1 sdr1-ctl.local',
        shell=True,
    ).decode()
    # Thu 20 Oct 2022 22:49:04.080492 CEST pinky-a:59660 STAT: (1) cmd tag 0 (2) commands 0 (3) Fri 21 Oct 2022 00:49:04.080382 CEST (16) out data src pinky-a:40023 (17) out data dst sdr1.local:5004 (18) out SSRC 1 666 298 019 (19) out TTL 1 (10) in samprate 2 400 000 Hz (22) out data pkts 30 497 405 (21) out metadata pkts 75 876 (69) output level -30,1 dB (24) calibration 0 (28) DC I offset 0 (68) gain 0,0 dB (33) RF 401 000 001,526 Hz (78) lock 1 (48) demod 0 (linear) (20) out samprate 2 400 000 Hz (49) out channels 2 (40) filt high 1,128e+06 Hz (39) filt low -1,128e+06 Hz (82) output bits/sample 8 (32) direct conv 1

    freq = output.split(' (33) RF ')[1].split(' Hz ', maxsplit=1)[0].replace('\u202f', '')
    sample_rate = output.split(' (20) out samprate ')[1].split(' Hz ', maxsplit=1)[0].replace('\u202f', '')
    return {
        'freq': int(float(freq.replace(',', '.'))),
        'sample_rate': int(sample_rate),
    }


def main():
    bin_size = 800
    crop = 0.25
    ci = get_channel_info()
    freq = ci['freq']
    sample_rate = ci['sample_rate']

    output = subprocess.check_output(
        '/home/pinky/work/air/ka9q-radio/iqrecord -S sdr1-ctl.local -d 1 -o -',
        shell=True,
    )
    now = datetime.utcnow().strftime('%Y-%m-%d, %H:%M:%S')
    k = sample_rate // bin_size
    chunk_size = 1
    while chunk_size < k:
        chunk_size *= 2
    data = np.frombuffer(output, dtype=np.int8)
    data_c = data.astype(np.float32).view(np.complex64) / 127
    power = np.zeros(chunk_size)
    nchunks = len(data_c) // chunk_size
    for n in range(0, nchunks):
        idx = n * chunk_size
        f = fft.fft(data_c[idx:idx+chunk_size])
        power += np.real(np.conj(f) * f)
    power = fft.fftshift(power)
    l = int((len(power) - len(power) * (1-crop)) // 2)
    power = power[l:-l]
    power /= nchunks * chunk_size
    p = 10 * np.log10(power)
    freq_lo = int(freq - sample_rate // 2 * (1-crop))
    freq_hi = int(freq + sample_rate // 2 * (1-crop))

    np.savetxt(
        'spectrum.txt',
        p,
        fmt='%.2f',
        newline=', ',
        header='%s, %d, %d, %.2f, %d' % (now, freq_lo, freq_hi, sample_rate/chunk_size, nchunks,),
        comments=''
    )


if __name__ == '__main__':
    main()
