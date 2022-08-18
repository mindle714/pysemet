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

def time_subalign(ref_data, reflen, deg_data, deglen,
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

def split_align(ref_data, reflen, ref_vad, ref_logvad,
                deg_data, deglen, deg_vad, deg_logvad,
                _utt_start, utt_speechstart, utt_speechend, _utt_end, 
                utt_delayest_1, _utt_delayconf, sr, searchbuf = 75):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  downsample = 32 if sr_mod == 'nb' else 64
  align_nfft = 512 if sr_mod == 'nb' else 1024
  window = 0.5 * (1 - np.cos((2 * np.pi * np.arange(align_nfft)) / align_nfft))

  uttlen = utt_speechend - utt_speechstart
  kernel = align_nfft / 64
  delta = align_nfft / (4 * downsample)
  step = math.floor((0.801 * uttlen + 40 * delta - 1) / (40 * delta))
  step *= delta

  pad = math.floor(uttlen / 10)
  pad = max(pad, searchbuf)

  utt_bps = np.zeros(41)
  utt_bps[0] = utt_speechstart + pad
  n_bps = 0

  while True:
    n_bps += 1
    utt_bps[n_bps] = utt_bps[n_bps - 1] + step
    if not (utt_bps[n_bps] <= (utt_speechend - pad) and n_bps < 40): break

  if n_bps <= 0: return

  utt_ed1 = np.zeros(41)
  utt_ed2 = np.zeros(41)

  for bp in range(n_bps):
    utt_ed1[bp] = crude_subalign(
      ref_logvad, reflen, deg_logvad, deglen,
      _utt_start, utt_bps[bp], utt_delayest_1, sr)

    utt_ed2[bp] = crude_subalign(
      ref_logvad, reflen, deg_logvad, deglen,
      utt_bps[bp], _utt_end, utt_delayest_1, sr)

  utt_dc1 = np.zeros(41) 
  utt_dc1[:n_bps] = -2.

  def _inner(startr, startd, h, hsum, utt_d1, utt_dc1):
    while (startd + align_nfft) <= deglen and \
      (startr + align_nfft) <= (utt_bps[bp] * downsample):
      x1 = ref_data[int(startr) : int(startr + align_nfft)] * window
      x2 = deg_data[int(startd) : int(startd + align_nfft)] * window

      x1_fft = np.fft.fft(x1, align_nfft)
      x1_fft_conj = np.conjugate(x1_fft)
      x2_fft = np.fft.fft(x2, align_nfft)
      x1 = np.fft.ifft(x1_fft_conj * x2_fft, align_nfft)

      x1 = np.abs(x1)
      v_max = np.max(x1) * 0.99
      n_max = (v_max ** 0.125) / kernel

      for count in range(align_nfft):
        if x1[count] > v_max:
          hsum += n_max * kernel
          for k in range(int(1 - kernel), int(kernel)):
            h[(count + k + align_nfft) % align_nfft] += n_max * (kernel - np.abs(k))

      startr += align_nfft / 4
      startd += align_nfft / 4

    i_max = np.argmax(h)
    v_max = h[i_max]
    if i_max >= (align_nfft / 2):
      i_max -= align_nfft

    utt_d1[bp] = estdelay + i_max
    if hsum > 0.:
      utt_dc1[bp] = v_max / hsum
    else:
      utt_dc1[bp] = 0.

    return startr, startd, h, hsum, utt_d1, utt_dc1

  def _inner2(startr, startd, h, hsum, utt_d2, utt_dc2):
    while startd >= 0 and startr >= utt_bps[bp] * downsample:
      x1 = ref_data[int(startr) : int(startr + align_nfft)] * window
      x2 = deg_data[int(startd) : int(startd + align_nfft)] * window
      
      x1_fft = np.fft.fft(x1, align_nfft)
      x1_fft_conj = np.conjugate(x1_fft)
      x2_fft = np.fft.fft(x2, align_nfft)
      x1 = np.fft.ifft(x1_fft_conj * x2_fft, align_nfft)
      
      x1 = np.abs(x1)
      v_max = np.max(x1) * 0.99
      n_max = (v_max ** 0.125) / kernel
      
      for count in range(align_nfft):
        if x1[count] > v_max:
          hsum += n_max * kernel
          for k in range(int(1 - kernel), int(kernel)):
            h[(count + k + align_nfft) % align_nfft] += n_max * (kernel - np.abs(k))
      
      startr -= align_nfft / 4
      startd -= align_nfft / 4
    
    i_max = np.argmax(h)
    v_max = h[i_max]
    if i_max >= (align_nfft / 2):
      i_max -= align_nfft
    
    utt_d2[bp] = estdelay + i_max
    if hsum > 0.:
      utt_dc2[bp] = v_max / hsum
    else:
      utt_dc2[bp] = 0.
    
    return startr, startd, h, hsum, utt_d2, utt_dc2

  utt_d1 = np.zeros(41)
  utt_d2 = np.zeros(41)

  # forward
  while True:
    bp = 0
    while bp < n_bps and utt_dc1[bp] > -2.: bp += 1
    if bp >= n_bps: break

    estdelay = utt_ed1[bp]
    h = np.zeros(align_nfft)
    hsum = 0.

    startr = _utt_start * downsample
    startd = startr + estdelay

    if startd < 0:
      startr = -estdelay
      startd = 0

    startr = max(startr, 0)
    startd = max(startd, 0)

    startr, startd, h, hsum, utt_d1, utt_dc1 = _inner(startr, startd, h, hsum, utt_d1, utt_dc1)
    while bp < (n_bps - 1):
      bp += 1
      if utt_ed1[bp] == estdelay and utt_dc1[bp] <= -2.:
        startr, startd, h, hsum, utt_d1, utt_dc1 = _inner(startr, startd, h, hsum, utt_d1, utt_dc1)
 
  utt_dc2 = np.zeros(41)
  for bp in range(n_bps - 1):
    if utt_dc1[bp] > _utt_delayconf:
      utt_dc2[bp] = -2.
    else:
      utt_dc2[bp] = 0.

  # backward
  while True:
    bp = n_bps - 1
    while bp >= 0 and utt_dc2[bp] > -2.: bp -= 1
    if bp < 0: break

    estdelay = utt_ed2[bp]
    h = np.zeros(align_nfft)
    hsum = 0.

    startr = _utt_end * downsample - align_nfft
    startd = startr + estdelay

    if (startd + align_nfft) > deglen:
      startd = deglen - align_nfft
      startr = startd - estdelay

    startr, startd, h, hsum, utt_d2, utt_dc2 = _inner2(startr, startd, h, hsum, utt_d2, utt_dc2)
    while bp > 0:
      bp -= 1
      if utt_ed2[bp] == estdelay and utt_dc2[bp] <= -2.:
        startr, startd, h, hsum, utt_d2, utt_dc2 = _inner2(startr, startd, h, hsum, utt_d2, utt_dc2)

  best_dc1 = 0.
  best_dc2 = 0.
  best_ed1 = 0
  best_ed2 = 0
  best_d1 = 0
  best_d2 = 0
  best_bp = 0

  for bp in range(n_bps):
    if np.abs(utt_d2[bp] - utt_d1[bp]) >= downsample and \
      (utt_dc1[bp] + utt_dc2[bp]) > (best_dc1 + best_dc2) and \
      utt_dc1[bp] > _utt_delayconf and utt_dc2[bp] > _utt_delayconf:
      best_ed1 = utt_ed1[bp]; best_d1 = utt_d1[bp]; best_dc1 = utt_dc1[bp]
      best_ed2 = utt_ed2[bp]; best_d2 = utt_d2[bp]; best_dc2 = utt_dc2[bp]
      best_bp = utt_bp[bp]

  return best_ed1, best_d1, best_dc1, best_ed2, best_d2, best_dc2, best_bp
