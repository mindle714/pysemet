import math
import numpy as np
np.set_printoptions(suppress=True)
from scipy import signal

def fftnxcorr(ref_vad, startr, nr, deg_vad, startd, nd):
  nx = 2 ** (math.ceil(math.log2(max(nr, nd))))
  x1 = np.zeros(2 * nx)
  x2 = np.zeros(2 * nx)
  startd = max(0, startd)
  startr = max(0, startr)

  x1[:nr] = ref_vad[startr : startr+nr][::-1]
  x2[:nd] = deg_vad[startd : startd+nd]

  x1_fft = np.fft.fft(x1, 2*nx)
  x2_fft = np.fft.fft(x2, 2*nx)
  tmp = np.fft.ifft(x1_fft * x2_fft, 2*nx)
  
  return tmp[:nr+nd-1]

def _crude_align(ref_logvad, deg_logvad, nr, nd, startr, startd):
  startr = max(0, int(startr))
  startd = max(0, int(startd))

  max_y = 0.
  i_max_y = nr - 1
  if nr > 1 and nd > 1:
    y = fftnxcorr(ref_logvad, startr, int(nr), deg_logvad, startd, int(nd))
    i_max_y = np.argmax(y)
    max_y = y[i_max_y]
    if max_y <= 0:
      max_y = 0
      i_max_y = nr - 1

  return max_y, i_max_y

def crude_align(ref_logvad, reflen, 
                deg_logvad, deglen, sr):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  downsample = 32 if sr_mod == 'nb' else 64 

  nr = math.floor(reflen / downsample)
  nd = math.floor(deglen / downsample)

  max_y, i_max_y = _crude_align(ref_logvad, deg_logvad, nr, nd, 0, 0)
  delayest = (i_max_y - (nr - 1)) * downsample
  return delayest

def crude_subalign(ref_logvad, reflen,
                   deg_logvad, deglen,
                   utt_start, utt_end, 
                   delayest, sr):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  downsample = 32 if sr_mod == 'nb' else 64 

  startr = utt_start
  startd = startr + delayest / downsample
  if startd < 0:
    startr = -delayest / downsample
    startd = 0

  nr = utt_end - startr
  nd = nr
  if (startd + nd) > math.floor(deglen / downsample):
    nd = math.floor(deglen / downsample) - startd

  max_y, i_max_y = _crude_align(ref_logvad, deg_logvad, nr, nd, startr, startd)
  return (i_max_y - (nr - 1)) * downsample + delayest

def crude_align_test(ref_logvad, deg_logvad, 
                     utt_start, utt_end, utt_delayest, 
                     delayest, sr):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  downsample = 32 if sr_mod == 'nb' else 64 

  startr = utt_start
  startd = startr + utt_delayest / downsample
  if startd < 0:
    startr = -utt_delayest / downsample
    startd = 0

  nr = utt_end - startr
  nd = nr
  if (startd + nd) > math.floor(deglen / downsample):
    nd = math.floor(deglen / downsample) - startd

  max_y, i_max_y = _crude_align(ref_logvad, deg_logvad, nr, nd, startr, startd)
  return (i_max_y - (nr - 1)) * downsample + delayest

def time_align(ref_data, reflen, deg_data, deglen,
               utt_start, utt_end, utt_delayest, sr):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  downsample = 32 if sr_mod == 'nb' else 64 
  align_nfft = 512 if sr_mod == 'nb' else 1024
  window = 0.5 * (1 - np.cos((2 * np.pi * np.arange(align_nfft)) / align_nfft))

  h = np.zeros(align_nfft)

  estdelay = utt_delayest
  startr = int(utt_start * downsample)
  startd = int(startr + estdelay)
  if startd < 0:
    startr = int(-estdelay)
    startd = 0

  while (startd + align_nfft) <= deglen and (startr + align_nfft) <= (utt_end * downsample):
    x1 = ref_data[startr : startr + align_nfft] * window
    x2 = deg_data[startd : startd + align_nfft] * window

    # cross-correlation between x1, x2
    x1_fft = np.fft.fft(x1, align_nfft)
    x1_fft_conj = np.conjugate(x1_fft)
    x2_fft = np.fft.fft(x2, align_nfft)
    x1 = np.fft.ifft(x1_fft_conj * x2_fft, align_nfft)

    x1 = np.abs(x1)
    v_max = np.max(x1) * 0.99

    h[x1 > v_max] += v_max ** 0.125
    startr += align_nfft // 4
    startd += align_nfft // 4

  x1 = h
  x2 = np.zeros(align_nfft)
  hsum = np.sum(h)

  x2[0] = 1.
  kernel = align_nfft // 64

  for count in range(1, kernel):
    x2[count] = 1 - count / kernel
    x2[-count] = 1 - count / kernel

  x1_fft = np.fft.fft(x1, align_nfft)
  x2_fft = np.fft.fft(x2, align_nfft)

  x1 = np.fft.ifft(x1_fft * x2_fft, align_nfft)

  if hsum > 0:
    h = np.abs(x1) / hsum
  else:
    h = 0

  i_max = np.argmax(h)
  v_max = h[i_max]
  if i_max >= (align_nfft / 2):
    i_max = i_max - align_nfft

  return estdelay + i_max, v_max
