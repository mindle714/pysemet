import math
import numpy as np
np.set_printoptions(suppress=True)
from scipy import signal

def xcorr(x1, x2, nfft):
  # cross-correlation between x1, x2
  x1_fft = np.fft.fft(x1, nfft)
  x1_fft_conj = np.conjugate(x1_fft)
  x2_fft = np.fft.fft(x2, nfft)
  x1 = np.fft.ifft(x1_fft_conj * x2_fft, nfft).real

  x1 = np.abs(x1)
  v_max = np.max(x1) * 0.99
  # integrity with c source
  #v_max = max(v_max, 0.)
  #v_max = np.max(np.abs(x1)) * 0.99
  return x1, v_max

def fftnxcorr(xf1, startr, nr, xf2, startd, nd):
  nx = 2 ** (math.ceil(math.log2(max(nr, nd))))
  x1 = np.zeros(2 * nx)
  x2 = np.zeros(2 * nx)
  startd = max(0, startd)
  startr = max(0, startr)

  x1[:nr] = xf1[startr : startr+nr][::-1]
  x2[:nd] = xf2[startd : startd+nd]

  x1_fft = np.fft.fft(x1, 2*nx)
  x2_fft = np.fft.fft(x2, 2*nx)
  tmp = np.fft.ifft(x1_fft * x2_fft, 2*nx).real
  
  return tmp[:nr+nd-1]

def _frame_align(rfdata, dfdata, nr, nd, startr, startd):
  startr = max(0, startr)
  startd = max(0, startd)

  max_y = 0.
  i_max_y = nr - 1
  if nr > 1 and nd > 1:
    y = fftnxcorr(rfdata, startr, nr, dfdata, startd, nd)
    i_max_y = np.argmax(y)
    max_y = y[i_max_y]
    if max_y <= 0:
      max_y = 0
      i_max_y = nr - 1

  return max_y, i_max_y

def frame_align_all(rfdata, nr, dfdata, nd):
  max_y, i_max_y = _frame_align(rfdata, dfdata, nr, nd, 0, 0)
  return i_max_y - (nr - 1)

def frame_align(rfdata, startr, endr, dfdata, nd, delayest):
  startd = startr + delayest
  if startd < 0:
    startr = - delayest
    startd = 0

  nr = endr - startr
  nd = min(nr, nd - startd)

  max_y, i_max_y = _frame_align(rfdata, dfdata, nr, nd, startr, startd)
  return i_max_y - (nr - 1)

def time_align(rdata, ddata, deglen,
               startr, endr, estdelay, window):
  startd = startr + estdelay
  if startd < 0:
    startr = -estdelay
    startd = 0

  winlen = window.shape[0]
  h = np.zeros(winlen)

  while (startd + winlen) <= deglen and (startr + winlen) <= endr:
    x1 = rdata[startr : startr + winlen] * window
    x2 = ddata[startd : startd + winlen] * window

    x1, v_max = xcorr(x1, x2, winlen)
    h[x1 > v_max] += v_max ** 0.125

    startr += winlen // 4
    startd += winlen // 4

  x1 = h
  x2 = np.zeros(winlen)
  hsum = np.sum(h)

  x2[0] = 1.
  kernel = winlen // 64

  for count in range(1, kernel):
    x2[count] = 1 - count / kernel
    x2[-count] = 1 - count / kernel

  x1_fft = np.fft.fft(x1, winlen)
  x2_fft = np.fft.fft(x2, winlen)
  x1 = np.fft.ifft(x1_fft * x2_fft, winlen).real

  if hsum > 0:
    h = np.abs(x1) / hsum
  else:
    h = np.zeros_like(x1)

  i_max = np.argmax(h)
  v_max = h[i_max]
  if i_max >= (winlen // 2):
    i_max -= winlen

  return estdelay + i_max, v_max
