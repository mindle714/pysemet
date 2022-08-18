import math
import numpy as np
np.set_printoptions(suppress=True)
from scipy import signal
import align

datapadding = 320

def nextpow2(x):
  return 1 if x == 0 else 2 ** math.ceil(math.log2(x))

def fix_power_level(data, datalen, maxlen, sr, searchbuf = 75):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  downsample = 32 if sr_mod == 'nb' else 64
  bufsamp = searchbuf * downsample
  padsamp = datapadding * (sr // 1000)

  filter_db = np.array([
    [0, 50, 100, 125, 160, 200, 250, 300, 350, 400, 
    500, 600, 630, 800, 1000, 1250, 1600, 2000, 2500, 3000, 
    3250, 3500, 4000, 5000, 6300, 8000],
    [-500, -500, -500, -500, -500, -500, -500, -500, 0, 0,  
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, -500, -500, -500, -500, -500]])

  aligned = apply_filter(data, datalen, filter_db, sr)
  pow_above_300 = np.sum(aligned[bufsamp : datalen - bufsamp + padsamp] ** 2)
  pow_above_300 /= (maxlen - 2 * bufsamp + padsamp)

  scale = np.sqrt(1e7 / pow_above_300)
  return data * scale

def apply_filter(data, datalen, filter_db, sr, searchbuf = 75):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  downsample = 32 if sr_mod == 'nb' else 64 
  bufsamp = searchbuf * downsample
  padsamp = datapadding * (sr // 1000)

  n = datalen - 2 * bufsamp + padsamp
  pow_of_2 = nextpow2(n)

  gainfilter = np.interp(1000, filter_db[0], filter_db[1])

  x = np.zeros(pow_of_2)
  x[:n] = data[bufsamp : bufsamp + n]
  x_fft = np.fft.fft(x, pow_of_2)

  freq_resolution = sr / pow_of_2
  factor_db = np.interp(np.arange(pow_of_2//2+1) * freq_resolution, 
    filter_db[0], filter_db[1])
  factor_db -= gainfilter

  factor = 10 ** (factor_db / 20)
  factor = np.concatenate([factor, factor[1:-1][::-1]], 0)
  x_fft = x_fft * factor

  y = np.fft.ifft(x_fft, pow_of_2)
  
  aligned = np.copy(data)
  aligned[bufsamp : bufsamp + n] = y[:n]
  return aligned

def dc_block(data, datalen, sr, searchbuf = 75):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  downsample = 32 if sr_mod == 'nb' else 64 
  bufsamp = searchbuf * downsample

  facc = np.sum(data[bufsamp : datalen - bufsamp]) / datalen
  mod_data = np.copy(data)
  mod_data[bufsamp : datalen-bufsamp] -= facc

  mod_data[bufsamp : bufsamp+downsample] *= (0.5 + np.arange(downsample)) / downsample
  mod_data[datalen-bufsamp-downsample : datalen-bufsamp] *= ((0.5 + np.arange(downsample)) / downsample)[::-1]

  return mod_data

def apply_filters(data, sr):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  if sr_mod == 'nb':
    iir_sos = np.array([
      [0.885535424, -0.885535424,  0.000000000, 1., -0.771070709,  0.000000000],
      [0.895092588,  1.292907193,  0.449260174, 1.,  1.268869037,  0.442025372],
      [4.049527940, -7.865190042,  3.815662102, 1., -1.746859852,  0.786305963],
      [0.500002353, -0.500002353,  0.000000000, 1.,  0.000000000,  0.000000000],
      [0.565002834, -0.241585934, -0.306009671, 1.,  0.259688659,  0.249979657],
      [2.115237288,  0.919935084,  1.141240051, 1., -1.587313419,  0.665935315],
      [0.912224584, -0.224397719, -0.641121413, 1., -0.246029464, -0.556720590],
      [0.444617727, -0.307589321,  0.141638062, 1., -0.996391149,  0.502251622]])

  else:
    iir_sos = np.array([
      [0.325631521, -0.086782860, -0.238848661, 1., -1.079416490,  0.434583902],
      [0.403961804, -0.556985881,  0.153024077, 1., -0.415115835,  0.696590244],
      [4.736162769,  3.287251046,  1.753289019, 1., -1.859599046,  0.876284034],
      [0.365373469,  0.000000000,  0.000000000, 1., -0.634626531,  0.000000000],
      [0.884811506,  0.000000000,  0.000000000, 1., -0.256725271,  0.141536777],
      [0.723593055, -1.447186099,  0.723593044, 1., -1.129587469,  0.657232737],
      [1.644910855, -1.817280902,  1.249658063, 1., -1.778403899,  0.801724355],
      [0.633692689, -0.284644314, -0.319789663, 1.,  0.000000000,  0.000000000],
      [1.032763031,  0.268428979,  0.602913323, 1.,  0.000000000,  0.000000000],
      [1.001616361, -0.823749013,  0.439731942, 1., -0.885778255,  0.000000000],
      [0.752472096, -0.375388990,  0.188977609, 1., -0.077258216,  0.247230734],
      [1.023700575,  0.001661628,  0.521284240, 1., -0.183867259,  0.354324187]])

  return signal.sosfilt(iir_sos, data)

def apply_filters_WB(data, sr):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  if sr_mod == 'nb':
    iir_sos = np.array([
      2.6657628, -5.3315255, 2.6657628, -1.8890331, 0.89487434])

  else:
    iir_sos = np.array([
      2.740826, -5.4816519, 2.740826, -1.9444777, 0.94597794])

  return signal.sosfilt(iir_sos, data)

def input_filter(ref_data, reflen, deg_data, deglen, sr):
  mod_ref_data = dc_block(ref_data, reflen, sr)
  mod_deg_data = dc_block(deg_data, deglen, sr)

  mod_ref_data = apply_filters(mod_ref_data, sr)
  mod_deg_data = apply_filters(mod_deg_data, sr)

  return mod_ref_data, mod_deg_data

def short_term_fft(Nf, data, window, start):
  x1 = data[int(start) : int(start + Nf)] * window
  x1_fft = np.fft.fft(x1)
  hz_spec = np.abs(x1_fft[:int(Nf/2)]) ** 2
  hz_spec[0] = 0
  return hz_spec

def freq_warping(hz_spectrum, nb, frame, sr):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  sp = 2.764344e-5 if sr_mod == 'nb' else 6.910853e-006 

  if sr_mod == 'nb':
    per_bark_band = np.array([
      1,    1,    1,    1,    1,     1,    1,    1,    2,    1,
      1,    1,    1,    1,    2,     1,    1,    2,    2,    2,
      2,    2,    2,    2,    2,     3,    3,    3,    3,    4,
      3,    4,    5,    4,    5,     6,    6,    7,    8,    9,
      9,    11
    ])
    pow_corr_factor = np.array([
        100.000000,  99.999992,   100.000000,  100.000008,   100.000008, 
        100.000015,  99.999992,   99.999969,   50.000027,    100.000000,     
        99.999969,   100.000015,  99.999947,   100.000061,   53.047077,     
        110.000046,  117.991989,  65.000000,   68.760147,    69.999931,     
        71.428818,   75.000038,   76.843384,   80.968781,    88.646126,     
        63.864388,   68.155350,   72.547775,   75.584831,    58.379192,     
        80.950836,   64.135651,   54.384785,   73.821884,    64.437073,     
        59.176456,   65.521278,   61.399822,   58.144047,    57.004543,     
        64.126297,   59.248363
    ])

  else:
    per_bark_band = np.array([
      1,    1,    1,    1,    1,   1,    1,    1,    2,    1,
      1,    1,    1,    1,    2,   1,    1,    2,    2,    2,
      2,    2,    2,    2,    2,   3,    3,    3,    3,    4,
      3,    4,    5,    4,    5,   6,    6,    7,    8,    9,
      9,    12,   12,   15,   16,  18,   21,   25,   20
    ])
    pow_corr_factor = np.array([
        100.000000,     99.999992,     100.000000,    100.000008,
        100.000008,     100.000015,    99.999992,     99.999969,  
        50.000027,      100.000000,    99.999969,     100.000015, 
        99.999947,      100.000061,    53.047077,     110.000046, 
        117.991989,     65.000000,     68.760147,     69.999931, 
        71.428818,      75.000038,     76.843384,     80.968781, 
        88.646126,      63.864388,     68.155350,     72.547775, 
        75.584831,      58.379192,     80.950836,     64.135651, 
        54.384785,      73.821884,     64.437073,     59.176456,     
        65.521278,      61.399822,     58.144047,     57.004543,     
        64.126297,      54.311001,     61.114979,     55.077751,     
        56.849335,      55.628868,     53.137054,     54.985844,    
        79.546974
    ])

  hz_band = 0
  pitch_pow_dens = np.zeros(nb)

  for bark_band in range(nb):
    n = per_bark_band[bark_band]

    _sum = np.sum(hz_spectrum[hz_band : hz_band + n])
    _sum *= pow_corr_factor[bark_band]
    _sum *= sp

    pitch_pow_dens[bark_band] = _sum
    hz_band += n

  return pitch_pow_dens

def get_abs_thresh_pow(sr):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  if sr_mod == 'nb':
    abs_thresh_pow = np.array([
        51286152,     2454709.500,  70794.593750,  
        4897.788574,  1174.897705,  389.045166,  
        104.712860,   45.708820,    17.782795,   
        9.772372,     4.897789,     3.090296,     
        1.905461,     1.258925,     0.977237,     
        0.724436,     0.562341,     0.457088,     
        0.389045,     0.331131,     0.295121,     
        0.269153,     0.257040,     0.251189,     
        0.251189,     0.251189,     0.251189,     
        0.263027,     0.288403,     0.309030,     
        0.338844,     0.371535,     0.398107,     
        0.436516,     0.467735,     0.489779,     
        0.501187,     0.501187,     0.512861,     
        0.524807,     0.524807,     0.524807
    ])

  else:
    abs_thresh_pow = np.array([
        51286152.00,  2454709.500,  70794.593750,  
        4897.788574,  1174.897705,  389.045166,     
        104.712860,   45.708820,    17.782795,    
        9.772372,     4.897789,     3.090296,   
        1.905461,     1.258925,     0.977237,     
        0.724436,     0.562341,     0.457088,     
        0.389045,     0.331131,     0.295121,     
        0.269153,     0.257040,     0.251189,    
        0.251189,     0.251189,     0.251189,    
        0.263027,     0.288403,     0.309030,     
        0.338844,     0.371535,     0.398107,    
        0.436516,     0.467735,     0.489779,    
        0.501187,     0.501187,     0.512861,    
        0.524807,     0.524807,     0.524807,    
        0.512861,     0.478630,     0.426580,    
        0.371535,     0.363078,     0.416869,    
        0.537032
    ])

  return abs_thresh_pow

def total_audible(frame, pitch_pow_dens, factor, sr):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  abs_thresh_pow = get_abs_thresh_pow(sr)
  Nb = 42 if sr_mod == 'nb' else 49 

  total_audible_pow = 0
  for band in range(1, Nb):
    h = pitch_pow_dens[frame, band]
    threshold = factor * abs_thresh_pow[band]
    if h > threshold:
      total_audible_pow += h

  return total_audible_pow

def time_sum_audible_of(silent, pitch_pow_dens, sr):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  abs_thresh_pow = get_abs_thresh_pow(sr)
  Nb = 42 if sr_mod == 'nb' else 49 

  avg_pitch_pow_dens = np.zeros(Nb)
  for band in range(Nb):
    result = 0
    for frame in range(silent.shape[0]):
      if not silent[frame]:
        h = pitch_pow_dens[frame, band]
        if h > (100 * abs_thresh_pow[band]):
          result += h
      avg_pitch_pow_dens[band] = result

  return avg_pitch_pow_dens

def freq_resp_compensation(nframes, pitch_pow_dens_ref, 
                           avg_pitch_pow_dens_ref, avg_pitch_pow_dens_deg, c, sr):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  Nb = 42 if sr_mod == 'nb' else 49 

  mod_pitch_pow_dens_ref = np.zeros_like(pitch_pow_dens_ref)

  for band in range(Nb):
    x = (avg_pitch_pow_dens_deg[band] + c) / (avg_pitch_pow_dens_ref[band] + c)
    x = max(min(x, 100.), 0.01)

    for frame in range(nframes):
      mod_pitch_pow_dens_ref[frame, band] = pitch_pow_dens_ref[frame, band] * x

  return mod_pitch_pow_dens_ref

def intensity_warping_of(frame, pitch_pow_dens, sr):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  abs_thresh_pow = get_abs_thresh_pow(sr)
  Nb = 42 if sr_mod == 'nb' else 49 
  sl = 1.866055e-1 if sr_mod == 'nb' else 1.866055e-001 

  if sr_mod == 'nb':
    center_of_band = np.array([
        0.078672,   0.316341,   0.636559,   0.961246,   1.290450, 
        1.624217,   1.962597,   2.305636,   2.653383,   3.005889, 
        3.363201,   3.725371,   4.092449,   4.464486,   4.841533, 
        5.223642,   5.610866,   6.003256,   6.400869,   6.803755, 
        7.211971,   7.625571,   8.044611,   8.469146,   8.899232, 
        9.334927,   9.776288,   10.223374,  10.676242,  11.134952,
        11.599563,  12.070135,  12.546731,  13.029408,  13.518232,
        14.013264,  14.514566,  15.022202,  15.536238,  16.056736,
        16.583761,  17.117382
    ])

  else:
    center_of_band = np.array([
        0.078672,   0.316341,   0.636559,    0.961246,     1.290450, 
        1.624217,   1.962597,   2.305636,    2.653383,     3.005889, 
        3.363201,   3.725371,   4.092449,    4.464486,     4.841533, 
        5.223642,   5.610866,   6.003256,    6.400869,     6.803755, 
        7.211971,   7.625571,   8.044611,    8.469146,     8.899232, 
        9.334927,   9.776288,   10.223374,   10.676242,    11.134952, 
        11.599563,  12.070135,  12.546731,   13.029408,    13.518232, 
        14.013264,  14.514566,  15.022202,   15.536238,    16.056736, 
        16.583761,  17.117382,  17.657663,   18.204674,    18.758478, 
        19.319147,  19.886751,  20.461355,   21.043034
    ])

  zwicker_pow = 0.23
  loudness_dens = np.zeros(Nb)

  for band in range(Nb):
    threshold = abs_thresh_pow[band]
    _input = pitch_pow_dens[frame, band]

    h = 1
    if center_of_band[band] < 4:
      h = min(6 / (center_of_band[band] + 2), 2)
    h = h ** 0.15

    mod_zwicker_pow = zwicker_pow * h
    if _input > threshold:
      loudness_dens[band] = ((threshold / 0.5) ** mod_zwicker_pow) * ((0.5 + 0.5 * _input / threshold) ** mod_zwicker_pow - 1)
    else:
      loudness_dens[band] = 0
    loudness_dens[band] = loudness_dens[band] * sl

  return loudness_dens

def id_searchwindows(ref_vad, reflen, deglen, 
                     delayest, sr, minutter = 50, searchbuf = 75):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  downsample = 32 if sr_mod == 'nb' else 64 

  utt_num = 1
  speech_flag = 0
  utt_starts = np.zeros(50)
  utt_ends = np.zeros(50)

  vadlen = math.floor(reflen / downsample)
  del_deg_start = minutter - delayest / downsample
  del_deg_end = math.floor((deglen - delayest) / downsample) - minutter

  for count in range(vadlen):
    vad_val = ref_vad[count]
    if vad_val > 0 and speech_flag == 0:
      speech_flag = 1
      this_start = count
      utt_starts[utt_num - 1] = max(count - searchbuf, 0)

    # bug exists in MATLAB pesq.m, where count == (vadlen - 2)
    if (vad_val == 0 or count == (vadlen - 1)) and speech_flag == 1:
      speech_flag = 0
      utt_ends[utt_num - 1] = min(count + searchbuf, vadlen - 1)

      if (count - this_start) >= minutter and this_start < del_deg_end and count > del_deg_start:
        utt_num += 1

  utt_num = utt_num - 1
  return utt_starts[:utt_num], utt_ends[:utt_num]

def id_utterances(utt_delay, ref_vad, reflen, deglen, 
                  delayest, sr, minutter = 50, searchbuf = 75):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  downsample = 32 if sr_mod == 'nb' else 64

  utt_starts, utt_ends = id_searchwindows(ref_vad, 
    reflen, deglen, delayest, sr, minutter = minutter, searchbuf = 0)

  utt_starts[0] = searchbuf
  nutter = utt_starts.shape[0]
  utt_ends[nutter - 1] = math.floor(reflen / downsample) - searchbuf

  for utt_num in range(1, nutter):
    this_start = utt_starts[utt_num]
    last_end = utt_ends[utt_num - 1]
    count = math.floor((this_start + last_end) / 2)
    utt_starts[utt_num] = count
    utt_ends[utt_num - 1] = count

  this_start = utt_starts[0] * downsample + utt_delay[0]
  if this_start < (searchbuf * downsample):
    count = searchbuf + math.floor((downsample - 1 - utt_delay[0]) / downsample)
    utt_starts[0] = count

  last_end = utt_ends[nutter - 1] * downsample + utt_delay[nutter - 1]
  if last_end > (deglen - searchbuf * downsample):
    count = math.floor((deglen - utt_delay[nutter - 1]) / downsample) - searchbuf
    utt_ends[nutter - 1] = count

  for utt_num in range(1, nutter):
    this_start = utt_starts[utt_num] * downsample + utt_delay[utt_num]
    last_end = utt_ends[utt_num - 1] * downsample + utt_delay[utt_num - 1]
    if this_start < last_end:
      count = math.floor((this_start + last_end) / 2)
      this_start = math.floor((downsample - 1 + count - utt_delay[utt_num]) / downsample)
      last_end = math.floor((count - utt_delay[utt_num - 1]) / downsample)
      utt_starts[utt_num] = this_start
      utt_ends[utt_num - 1] = last_end

  return utt_starts, utt_ends

def utterance_locate(ref_data, reflen, ref_vad, ref_logvad,
                     deg_data, deglen, deg_vad, deg_logvad, delayest, sr):

  uttsearch_starts, uttsearch_ends = id_searchwindows(
    ref_vad, reflen, deglen, delayest, sr)
  nutter = uttsearch_starts.shape[0]

  utt_delayest = np.zeros(50)
  utt_delay = np.zeros(50)
  utt_delayconf = np.zeros(50)

  for utt_id in range(nutter):
    utt_delayest[utt_id] = align.crude_subalign(
      ref_logvad, reflen, deg_logvad, deglen,
      uttsearch_starts[utt_id], uttsearch_ends[utt_id], delayest, sr)

    _utt_delay, _utt_delayconf = align.time_subalign(
      ref_data, reflen, deg_data, deglen,
      uttsearch_starts[utt_id], uttsearch_ends[utt_id], utt_delayest[utt_id], sr)
    utt_delay[utt_id] = _utt_delay
    utt_delayconf[utt_id] = _utt_delayconf

  utt_starts, utt_ends = id_utterances(utt_delay, ref_vad, reflen, deglen, delayest, sr)
 
  # utterance split
  utt_id = 0
  while utt_id < nutter and nutter <= 50:
    _utt_start = utt_starts[utt_id]
    _utt_end = utt_ends[utt_id]

    # adjust utterance start, end based on VAD
    utt_speechstart = max(0, _utt_start)
    while utt_speechstart < (_utt_end-1) and ref_vad[int(utt_speechstart)] <= 0:
      utt_speechstart += 1

    utt_speechend = _utt_end
    while utt_speechend > _utt_start and ref_vad[int(utt_speechend)] <= 0:
      utt_speechend -= 1
    utt_speechend += 1
    uttlen = utt_speechend - utt_speechstart

    if uttlen >= 200:
      _utt_delayconf = utt_delayconf[utt_id]
      best_ed1, best_d1, best_dc1, best_ed2, best_d2, best_dc2, best_bp = align.split_align(
        ref_data, reflen, ref_vad, ref_logvad, 
        deg_data, deglen, deg_vad, deg_logvad,
        _utt_start, utt_speechstart, utt_speechend, _utt_end, 
        utt_delayest[utt_id], _utt_delayconf, sr)

      if best_dc1 > _utt_delayconf and best_dc2 > _utt_delayconf:
        for step in range(nutter - 1, utt_id, -1):
          utt_delayest[step + 1] = utt_delayest[step]
          utt_delay[step + 1] = utt_delay[step]
          utt_delayconf[step + 1] = utt_delayconf[step]
          utt_starts[step + 1] = utt_starts[step]
          utt_ends[step + 1] = utt_ends[step]
          utt_starts[step + 1] = utt_starts[step]
          utt_ends[step + 1] = utt_ends[step]

        nutter += 1

        utt_delayest[utt_id] = best_ed1
        utt_delay[utt_id] = best_d1
        utt_delayconf[utt_id] = best_dc1

        utt_delayest[utt_id + 1] = best_ed2
        utt_delay[utt_id + 1] = best_d2
        utt_delayconf[utt_id + 1] = best_dc2
        utt_starts[utt_id + 1] = utt_starts[utt_id]
        utt_ends[utt_id + 1] = utt_ends[utt_id]

        if best_d2 < best_d1:
          utt_starts[utt_id] = _utt_start
          utt_ends[utt_id] = best_bp
          utt_starts[utt_id + 1] = best_bp
          utt_ends[utt_id + 1] = _utt_end
        else:
          utt_starts[utt_id] = _utt_start
          utt_ends[utt_id] = best_bp + math.floor((best_d2 - best_d1) / (2 * downsample)) 
          utt_starts[utt_id + 1] = best_bp - math.floor((best_d2 - best_d1) / (2 * downsample))
          utt_ends[utt_id + 1] = _utt_end

        if ((utt_starts[utt_id] - searchbuf) * downsample + best_d1) < 0:
          utt_starts[utt_id] = searchbuf + math.floor((downsample - best_d1) / downsample)

        if (utt_ends[utt_id + 1] * downsample + best_d2) > (deglen - searchbuf * downsample):
          utt_ends[utt_id + 1] = math.floor((deglen - best_d2) / downsample) - searchbuf

      else:
        utt_id += 1

    else:
      utt_id += 1

  return utt_starts, utt_ends, utt_delay, nutter
  
def pseudo_lp(x, p, sr):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  Nb = 42 if sr_mod == 'nb' else 49 

  if sr_mod == 'nb':
    width_of_band = np.array([
        0.157344,     0.317994,     0.322441,     0.326934,     0.331474,     
        0.336061,     0.340697,     0.345381,     0.350114,     0.354897,     
        0.359729,     0.364611,     0.369544,     0.374529,     0.379565,     
        0.384653,     0.389794,     0.394989,     0.400236,     0.405538,     
        0.410894,     0.416306,     0.421773,     0.427297,     0.432877,     
        0.438514,     0.444209,     0.449962,     0.455774,     0.461645,     
        0.467577,     0.473569,     0.479621,     0.485736,     0.491912,     
        0.498151,     0.504454,     0.510819,     0.517250,     0.523745,     
        0.530308,     0.536934
    ])

  else:
    width_of_band = np.array([
        0.157344,     0.317994,     0.322441,     0.326934,     0.331474,
        0.336061,     0.340697,     0.345381,     0.350114,     0.354897,
        0.359729,     0.364611,     0.369544,     0.374529,     0.379565,
        0.384653,     0.389794,     0.394989,     0.400236,     0.405538,
        0.410894,     0.416306,     0.421773,     0.427297,     0.432877,
        0.438514,     0.444209,     0.449962,     0.455774,     0.461645,
        0.467577,     0.473569,     0.479621,     0.485736,     0.491912,
        0.498151,     0.504454,     0.510819,     0.517250,     0.523745,
        0.530308,     0.536934,     0.543629,     0.550390,     0.557220,
        0.564119,     0.571085,     0.578125,     0.585232
    ])

  result = 0
  tot = 0

  for band in range(1, Nb):
    h = np.abs(x[band])
    w = width_of_band[band]
    _prod = h * w

    result += _prod ** p
    tot += w

  result = (result / tot) ** (1/p)
  result *= tot
  return result

def multiply_with_asymm_factor(disturbance_dens, frame, pitch_pow_dens_ref, pitch_pow_dens_deg, sr):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  Nb = 42 if sr_mod == 'nb' else 49 

  mod_disturbance_dens = np.zeros(Nb)
  for i in range(Nb):
    ratio = (pitch_pow_dens_deg[frame, i] + 50) / (pitch_pow_dens_ref[frame, i] + 50)
    h = ratio ** 1.2
    h = min(h, 12)
    if h < 3:
      h = 0
    mod_disturbance_dens[i] = disturbance_dens[i] * h

  return mod_disturbance_dens

def compute_delay(start, stop, search_range, t1, t2): 
  n = stop - start
  pow_of_2 = 2 ** math.ceil(math.log2(2 * n))

  pow1 = np.mean(t1[start : stop] ** 2) * (n / pow_of_2)
  pow2 = np.mean(t2[start : stop] ** 2) * (n / pow_of_2)
  normalization = np.sqrt(pow1 * pow2)

  if pow1 <= 1e-6 or pow2 <= 1e-6:
    max_corr = 0
    best_delay = 0

  x1 = np.zeros(pow_of_2)
  x2 = np.zeros(pow_of_2)

  x1[:n] = np.abs(t1[start:stop])
  x2[:n] = np.abs(t2[start:stop])

  x1_fft = np.fft.fft(x1, pow_of_2) / pow_of_2
  x2_fft = np.fft.fft(x2, pow_of_2)
  x1_fft_conj = np.conjugate(x1_fft)
  y = np.fft.ifft(x1_fft_conj * x2_fft, pow_of_2)

  best_delay = 0
  max_corr = 0

  for i in range(-search_range, -1):
    h = np.abs(y[i + pow_of_2]) / normalization
    if h > max_corr:
      max_corr = h
      best_delay = i

  for i in range(search_range):
    h = np.abs(y[i]) / normalization
    if h > max_corr:
      max_corr = h
      best_delay = i

  best_delay -= 1
  return best_delay, max_corr

def lpq_weight(start, stop, pow_syl, pow_time, 
               frame_disturbance, tw, per_syl = 20):
  result_time = 0
  tot_time = 0
  for start_syl in range(start, stop + 1, (per_syl//2)):
    result_syl = 0
    count_syl = 0

    for frame in range(start_syl, start_syl + per_syl):
      if frame <= stop:
        h = frame_disturbance[frame]
        result_syl += h ** pow_syl
      count_syl += 1

    result_syl /= count_syl
    result_syl **= (1 / pow_syl)

    result_time += (tw[start_syl - start] * result_syl) ** pow_time
    tot_time += tw[start_syl - start] ** pow_time

  result_time /= tot_time
  result_time **= (1 / pow_time)
  return result_time

def process_bad(frame_disturbance):
  stop_frame = frame_disturbance.shape[0] - 1

  frame_bad = [(fd > threshold_bad_frame) for frame in frame_disturbance]
  frame_bad[0] = False

  smear_range = 2

  smear_frame_bad = [False for _ in range(stop_frame + 1)]
  for frame in range(smear_range, stop_frame - smear_range):
    max_left = np.max(frame_bad[frame - smear_range : frame])
    max_right = np.max(frame_bad[frame : frame + smear_range])
    smear_frame_bad[frame] = min(max_left, max_right)

  minimum_bad_frame_in_interval = 5
  num_bad_interval = 0
  frame = 0

  while frame <= stop_frame:
    while frame <= stop_frame and not smear_frame_bad[frame]:
      frame += 1

    if frame <= stop_frame:
      start_frame_bad_interval[num_bad_interval] = frame
      while frame <= stop_frame and smear_frame_bad[frame]:
        frame += 1

      if frame <= stop_frame:
        stop_frame_bad_interval[num_bad_interval] = frame
        if (stop_frame_bad_interval[num_bad_interval] - start_frame_bad_interval[num_bad_interval]) >= minimum_bad_frame_in_interval:
          num_bad_interval += 1

  for bad_interval in range(num_bad_interval):
    start_samp_bad_interval[bad_interval] = (start_frame_bad_interval[bad_interval] * (Nf / 2)) + bufsamp
    stop_samp_bad_interval[bad_interval] = (stop_frame_bad_interval[bad_interval] * (Nf / 2)) + bufsamp
    if stop_frame_bad_interval[bad_interval] > stop_frame:
      stop_frame_bad_interval[bad_interval] = stop_frame

    num_samp_bad_interval[bad_interval] = stop_samp_bad_interval[bad_interval] - start_samp_bad_interval[bad_interval]

  search_range = 4
  search_range_samp = search_range * Nf

  for bad_interval in range(num_bad_interval):
    num_samp = num_samp_bad_interval[bad_interval]
    ref = np.zeros(2 * search_range_samp + num_samp)
    deg = np.zeros(2 * search_range_samp + num_samp)

    start_samp = start_samp_bad_interval[bad_interval]
    ref[search_range_samp:][:num_samp] = ref_data[start_samp:][:num_samp]

    for i in range(deg.shape[0]):
      j = start_samp - search_range + i
      nn = maxlen - bufsamp + padsamp
      j = max(min(j, nn), bufsamp)
      deg[i] = tweaked_deg[j]

    delay_samp, best_corr = compute_delay(0, 2 * search_range_samp + num_samp, search_range_samp, ref, deg)
    delay_samp_bad_interval[bad_interval] = delay_samp
    if best_corr < 0.5:
      delay_samp_bad_interval[bad_interval] = 0

  if num_bad_interval > 0:
    doubly_tweaked_deg = tweaked_deg[:maxlen + padsamp]
    for bad_interval in range(num_bad_interval):
      delay = delay_samp_bad_interval[bad_interval]

      for i in range(start_samp_bad_interval[bad_interval], stop_samp_bad_interval[bad_interval]):
        j = i + delay
        j = max(min(j, maxlen), 0)
        doubly_tweaked_deg[i] = tweaked_deg[j]

      untweaked_deg = deg_data
      deg_data = doubly_tweaked_deg

      for bad_interval in range(num_bad_interval):
        for frame in range(start_frame_bad_interval[bad_interval], stop_frame_bad_interval[bad_interval]):
          start_samp_ref = searchbuf * downsample + (frame * Nf / 2)
          start_samp_deg = start_samp_ref
          hz_deg = short_term_fft(Nf, deg_data, window, start_samp_deg)
          pitch_pow_dens_deg[frame, :] = freq_warping(hz_deg, Nb, frame)

        old_scale = 1
        for frame in range(start_frame_bad_interval[bad_interval], stop_frame_bad_interval[bad_interval]):
          total_audible_pow_ref = total_audible(frame, pitch_pow_dens_ref, 1)
          total_audible_pow_deg = total_audible(frame, pitch_pow_dens_deg, 1)
          scale = (total_audible_pow_ref + 5e3) / (total_audible_pow_deg + 5e3)

          if frame > 0:
            scale = 0.2 * old_scale + 0.8 * scale
          old_scale = scale
          scale = max(min(scale, max_scale), min_scale)

          pitch_pow_dens_deg[frame, :] *= scale
          loudness_dens_ref = intensity_warping_of(frame, pitch_pow_dens_ref) 
          loudness_dens_deg = intensity_warping_of(frame, pitch_pow_dens_deg)
          disturbance_dens = loudness_dens_deg - loudness_dens_ref

          for band in range(Nb):
            m = min(loudness_dens_deg[band], loudness_dens_ref[band]) * 0.25
            d = disturbance_dens[band]

            if d > m:
              disturbance_dens[band] -= m
            else:
              if d < -m:
                disturbance_dens[band] += m
              else:
                disturbance_dens[band] = 0

          frame_disturbance[frame] = min(frame_disturbance[frame], pseudo_lp(disturbance_dens, d_pow_f))
          disturbance_dens = multiply_with_asymm_factor(disturbance_dens, frame, pitch_pow_dens_ref, pitch_pow_dens_deg)
          frame_disturbance_asym_add[frame] = min(frame_disturbance_asym_add[frame], pseudo_lp(disturbance_dens, a_pow_f))
            
      deg_data = untweaked_deg   

def pesq_psychoacoustic_model(ref_data, reflen, deg_data, deglen, sr,
                              utt_starts, utt_ends, utt_delay, nutter, 
                              searchbuf = 75, silence_5samp = 500):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  downsample = 32 if sr_mod == 'nb' else 64 
  bufsamp = searchbuf * downsample
  padsamp = datapadding * (sr // 1000)

  Nf = downsample * 8
  window = 0.5 * (1 - np.cos(2 * np.pi * (np.arange(Nf) / Nf)))
  Nb = 42 if sr_mod == 'nb' else 49 
  
  maxlen = max(reflen, deglen)

  # trim both side
  skip_start = 0
  while True:
    pos = skip_start + bufsamp
    samp_sum = np.sum(np.abs(ref_data[pos : pos + 5]))
    if samp_sum < silence_5samp: skip_start += 1
    if not ((samp_sum < silence_5samp) and (skip_start < maxlen / 2)): break
  
  skip_end = 0
  while True:
    pos = maxlen - bufsamp + padsamp - 5 - skip_end
    samp_sum = np.sum(np.abs(ref_data[pos : pos + 5]))
    if samp_sum < silence_5samp: skip_end += 1
    if not ((samp_sum < silence_5samp) and (skip_end < maxlen / 2)): break

  start_frame = math.floor(skip_start / (Nf / 2))
  stop_frame = math.floor((maxlen - 2 * bufsamp + padsamp - skip_end) / (Nf / 2)) - 1

  power_ref = np.mean(ref_data[bufsamp : maxlen - bufsamp + padsamp] ** 2)
  power_deg = np.mean(deg_data[bufsamp : maxlen - bufsamp + padsamp] ** 2)

  pitch_pow_dens_ref = np.zeros((stop_frame + 1, Nb))
  pitch_pow_dens_deg = np.zeros((stop_frame + 1, Nb))
  silent = np.zeros(stop_frame + 1)
  total_power_ref = np.zeros(stop_frame + 1)

  frame_disturbance = np.zeros(stop_frame + 1)
  frame_disturbance_asym_add = np.zeros(stop_frame + 1)
  tw = np.zeros(stop_frame + 1)

  for frame in range(stop_frame + 1):
    start_sample_ref = bufsamp + frame * Nf / 2 
    hz_ref = short_term_fft(Nf, ref_data, window, start_sample_ref)

    utt = nutter - 1
    while utt >= 0 and (utt_starts[utt] * downsample) > start_sample_ref:
      utt -= 1

    delay = utt_delay[max(utt, 0)]
    start_sample_deg = start_sample_ref + delay

    if start_sample_deg > 0 and (start_sample_deg + Nf) < (maxlen + padsamp):
      hz_deg = short_term_fft(Nf, deg_data, window, start_sample_deg)
    else:
      hz_deg[:Nf/2] = 0

    pitch_pow_dens_ref[frame, :] = freq_warping(hz_ref, Nb, frame, sr)
    pitch_pow_dens_deg[frame, :] = freq_warping(hz_deg, Nb, frame, sr)

    total_audible_pow_ref = total_audible(frame, pitch_pow_dens_ref, 1e2, sr)
    total_audible_pow_deg = total_audible(frame, pitch_pow_dens_deg, 1e2, sr)
    silent[frame] = total_audible_pow_ref < 1e7

  denom = math.floor((maxlen - 2 * bufsamp + padsamp) / (Nf / 2)) - 1
  avg_pitch_pow_dens_ref = time_sum_audible_of(silent, pitch_pow_dens_ref, sr) / denom
  avg_pitch_pow_dens_deg = time_sum_audible_of(silent, pitch_pow_dens_deg, sr) / denom

  calibrate = False
  if not calibrate:
    pitch_pow_dens_ref = freq_resp_compensation(stop_frame + 1,
      pitch_pow_dens_ref, avg_pitch_pow_dens_ref, avg_pitch_pow_dens_deg, 1000, sr)

  d_pow_f = 2
  d_pow_s = 6
  d_pow_t = 2
  a_pow_f = 1
  a_pow_s = 6
  a_pow_t = 2

  max_scale = 5.
  min_scale = 3e-4
  old_scale = 1
  threshold_bad_frames = 30
  bad_frame = False

  for frame in range(stop_frame + 1):
    total_audible_pow_ref = total_audible(frame, pitch_pow_dens_ref, 1, sr)
    total_audible_pow_deg = total_audible(frame, pitch_pow_dens_deg, 1, sr)
    total_power_ref[frame] = total_audible_pow_ref

    scale = (total_audible_pow_ref + 5e3) / (total_audible_pow_deg + 5e3)
    if frame > 0:
      scale = 0.2 * old_scale + 0.8 * scale
    old_scale = scale

    scale = max(min(scale, max_scale), min_scale)
    pitch_pow_dens_deg[frame, :] *= scale

    loudness_dens_ref = intensity_warping_of(frame, pitch_pow_dens_ref, sr)
    loudness_dens_deg = intensity_warping_of(frame, pitch_pow_dens_deg, sr)
    disturbance_dens = loudness_dens_deg - loudness_dens_ref

    for band in range(Nb):
      m = 0.25 * min(loudness_dens_deg[band], loudness_dens_ref[band])
      d = disturbance_dens[band]

      if d > m:
        disturbance_dens[band] -= m
      else:
        if d < -m:
          disturbance_dens[band] += m
        else:
          disturbance_dens[band] = 0

    frame_disturbance[frame] = pseudo_lp(disturbance_dens, d_pow_f, sr)
    if frame_disturbance[frame] > threshold_bad_frames:
      bad_frame = True

    disturbance_dens = multiply_with_asymm_factor(disturbance_dens, frame, pitch_pow_dens_ref, pitch_pow_dens_deg, sr)
    frame_disturbance_asym_add[frame] = pseudo_lp(disturbance_dens, a_pow_f, sr)

  frame_skipped = [False for _ in range(stop_frame + 1)]
  for utt in range(1, nutter):
    frame1 = math.floor(((utt_starts[utt] - searchbuf) * downsample + utt_delay[utt]) / (Nf / 2))
    j = math.floor(math.floor((utt_ends[utt - 1] - searchbuf) * downsample + utt_delay[utt - 1]) / (Nf / 2))
    delay_jump = utt_delay[utt] - utt_delay[utt - 1]
    frame1 = max(min(frame1, j), 0)

    if delay_jump < -(Nf / 2):
      frame2 = math.floor(((utt_starts[utt] - searchbuf) * downsample + max(0, np.abs(delay_jump))) / (Nf / 2)) + 1
      for frame in range(frame1, frame2 + 1):
        if frame < stop_frame:
          frame_skipped[frame] = True
          frame_disturbance[frame] = 0
          frame_disturbance_asym_add[frame] = 0

  nn = padsamp + maxlen
  tweaked_deg = np.zeros(nn)
  for i in range(bufsamp, nn - bufsamp):
    utt = nutter - 1
    while utt >= 0 and (utt_starts[utt] * downsample) > i:
      utt -= 1

    if utt >= 0:
      delay = utt_delay[utt]
    else:
      delay = utt_delay[0]

    j = i + delay
    j = max(min(j, nn - bufsamp - 1), bufsamp)
    tweaked_deg[i] = deg_data[int(j)]

  if bad_frame:
    process_bad()

  for frame in range(stop_frame + 1):
    h = 1
    if (stop_frame + 1) > 1000:
      n = math.floor((maxlen - 2 * searchbuf * downsample) / (Nf / 2)) - 1
      twf = min((n - 1000) / 5500, 0.5)
      h = (1. - twf) + twf * frame / n

    tw[frame] = h
  
  for frame in range(stop_frame + 1):
    h = ((total_power_ref[frame] + 1e5) / 1e7) ** 0.04
    frame_disturbance[frame] /= h
    frame_disturbance_asym_add[frame] /= h
    frame_disturbance[frame] = min(frame_disturbance[frame], 45)
    frame_disturbance_asym_add[frame] = min(frame_disturbance_asym_add[frame], 45)

  d_ind = lpq_weight(start_frame, stop_frame, d_pow_s, d_pow_t, frame_disturbance, tw)
  a_ind = lpq_weight(start_frame, stop_frame, a_pow_s, a_pow_t, frame_disturbance_asym_add, tw)

  d_weight = 0.1
  a_weight = 0.0309
  pesq_mos = 4.5 - d_weight * d_ind - a_weight * a_ind
  return pesq_mos

def apply_vad(data, datalen, sr,
              minspeech=4, joinspeech=50):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  downsample = 32 if sr_mod == 'nb' else 64

  nwindows = math.floor(datalen / downsample)
  vad = np.zeros(nwindows)
  for count in range(nwindows):
    vad[count] = np.mean(data[count * downsample : (count+1) * downsample] ** 2)

  level_thres = np.mean(vad)
  level_min = np.max(vad)
  if level_min > 0:
    level_min *= 1e-4
  else:
    level_min = 1.

  vad[vad<level_min] = level_min

  for iteration in range(12):
    level_noise = 0
    std_noise = 0

    vad_less = vad[vad <= level_thres]
    level_noise = np.sum(vad_less)
    if vad_less.shape[0] > 0:
      level_noise /= vad_less.shape[0]
      std_noise = np.sqrt(np.mean((vad_less - level_noise) ** 2))
    level_thres = 1.001 * (level_noise + 2 * std_noise)
    
  level_noise = 0
  vad_greater = vad[vad > level_thres]
  level_sig = np.sum(vad_greater)

  vad_lesseq = vad[vad <= level_thres]
  level_noise = np.sum(vad_lesseq)

  if vad_greater.shape[0] > 0:
    level_sig /= vad_greater.shape[0]
  else:
    level_thres = -1

  if vad_greater.shape[0] < nwindows:
    level_noise /= (nwindows - vad_greater.shape[0])
  else:
    level_noise = 1

  vad[vad <= level_thres] = -vad[vad <= level_thres]
  vad[0] = -level_min
  vad[-1] = -level_min

  start = 0
  finish = 0
  for count in range(1, nwindows):
    if vad[count] > 0. and vad[count-1] <= 0.:
      start = count
    if vad[count] <= 0. and vad[count-1] > 0.:
      finish = count
      if (finish - start) <= minspeech:
        vad[start : finish] *= -1

  if level_sig >= (level_noise * 1000):
    for count in range(1, nwindows):
      if vad[count] > 0 and vad[count-1] <= 0:
        start = count
      if vad[count] <= 0 and vad[count-1] > 0:
        finish = count
        g = np.sum(vad[start : finish])
        if g < (3. * level_thres * (finish - start)):
          vad[start : finish] *= -1

  start = 0
  finish = 0
  for count in range(1, nwindows):
    if vad[count] > 0. and vad[count-1] <= 0.:
      start = count
      if finish > 0 and (start - finish) <= joinspeech:
        vad[finish : start] = level_min
    if vad[count] <= 0. and vad[count-1] > 0.:
      finish = count

  start = 0
  for count in range(1, nwindows):
    if vad[count] > 0 and vad[count-1] <= 0:
      start = count

  if start == 0:
    vad = np.abs(vad)
    vad[0] = -level_min
    vad[-1] = -level_min

  count = 3
  while count < (nwindows - 2):
    if vad[count] > 0 and vad[count-2] <= 0:
      vad[count-2] = vad[count] * 0.1
      vad[count-1] = vad[count] * 0.3
      count += 1
    if vad[count] <= 0 and vad[count-1] > 0:
      vad[count] = vad[count-1] * 0.3
      vad[count+1] = vad[count-1] * 0.1
      count += 3
    count += 1

  vad[vad < 0] = 0
  if level_thres <= 0:
    level_thres = level_min

  log_vad = np.zeros(nwindows)
  log_vad[vad <= level_thres] = 0
  log_vad[vad > level_thres] = np.log(vad[vad > level_thres] / level_thres)

  return vad, log_vad

def pesq(ref_data, deg_data, sr, searchbuf = 75):
  assert sr == 16000 or sr == 8000
  sr_mod = 'wb' if sr == 16000 else 'nb'
  downsample = 32 if sr_mod == 'nb' else 64
  bufsamp = searchbuf * downsample
  padsamp = datapadding * (sr // 1000)
  
  winlength = 512 if sr_mod == 'nb' else 1024
  window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(winlength)/winlength))
  
  reflen = ref_data.shape[0] + 2 * bufsamp
  deglen = deg_data.shape[0] + 2 * bufsamp
  maxlen = max(reflen, deglen)

  # pad zeros on ref, deg pcms
  ref_data = np.concatenate([
    np.zeros(bufsamp), ref_data * 32768, np.zeros(bufsamp + padsamp)])
  deg_data = np.concatenate([
    np.zeros(bufsamp), deg_data * 32768, np.zeros(bufsamp + padsamp)])

  ref_data = fix_power_level(ref_data, reflen, maxlen, sr)
  deg_data = fix_power_level(deg_data, deglen, maxlen, sr)

  if sr_mod == 'nb':
    IRS_filter_db = np.array([
      [0, 50, 100, 125, 160, 200, 250, 300, 350, 400, 
      500, 600, 700, 800, 1000, 1300, 1600, 2000, 2500, 3000, 
      3250, 3500, 4000, 5000, 6300, 8000], 
      [-200, -40, -20, -12, -6, 0, 4, 6, 8, 10, 
      11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 
      12, 4, -200, -200, -200, -200]])

    ref_data = apply_filter(ref_data, reflen, IRS_filter_db, sr)
    deg_data = apply_filter(deg_data, deglen, IRS_filter_db, sr)

  else:
    ref_data = apply_filters_WB(ref_data)
    deg_data = apply_filters_WB(deg_data)

  model_ref = ref_data
  model_deg = deg_data

  ref_data, deg_data = input_filter(ref_data, reflen, deg_data, deglen, sr)
  ref_vad, ref_logvad = apply_vad(ref_data, reflen, sr)
  deg_vad, deg_logvad = apply_vad(deg_data, deglen, sr)

  delayest = align.crude_align(ref_logvad, reflen, deg_logvad, deglen, sr)
  utt_starts, utt_ends, utt_delay, nutter = utterance_locate(
    ref_data, reflen, ref_vad, ref_logvad, 
    deg_data, deglen, deg_vad, deg_logvad, delayest, sr)

  ref_data = model_ref
  deg_data = model_deg

  pesq_mos = pesq_psychoacoustic_model(ref_data, reflen, deg_data, deglen, sr,
    utt_starts, utt_ends, utt_delay, nutter)

  if sr_mod == 'nb':
    mapped = 0.999 + 4. / (1. + np.exp(-1.4945 * pesq_mos + 4.6607))
    return pesq_mos, mapped

  else:
    mapped = 0.999 + 4. / (1. + np.exp(-1.3669 * pesq_mos + 3.8224))
    return mapped