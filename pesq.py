import math
import numpy as np
np.set_printoptions(suppress=True)
from scipy import signal
import align
import pesq_model

datapadding = 320

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
  pow_of_2 = 2 ** math.ceil(math.log2(n))

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

def id_searchwindows(ref_vad, vadlen, deglen, 
                     delayest, minutter = 50, searchbuf = 75):
  speech_flag = 0
  utt_starts = np.zeros(50)
  utt_ends = np.zeros(50)

  del_deg_start = minutter - delayest
  del_deg_end = deglen - minutter

  utt_num = 0; this_start = 0
  for count in range(vadlen):
    vad_val = ref_vad[count]
    if vad_val > 0 and speech_flag == 0:
      speech_flag = 1
      this_start = count
      utt_starts[utt_num] = max(count - searchbuf, 0)

    # bug exists in MATLAB pesq.m, where count == (vadlen - 2)
    if (vad_val == 0 or count == (vadlen - 1)) and speech_flag == 1:
      speech_flag = 0
      utt_ends[utt_num] = min(count + searchbuf, vadlen - 1)

      if (count - this_start) >= minutter and this_start < del_deg_end and count > del_deg_start:
        utt_num += 1

  return utt_starts[:utt_num], utt_ends[:utt_num]

def id_utterances(utt_delay, ref_vad, reflen, deglen, 
                  delayest, sr, minutter = 50, searchbuf = 75):
  sr_mod = 'wb' if sr == 16000 else 'nb'
  downsample = 32 if sr_mod == 'nb' else 64
  bufsamp = searchbuf * downsample

  reflen = math.floor(reflen / downsample)

  utt_starts, utt_ends = id_searchwindows(ref_vad, reflen, 
    math.floor((deglen - delayest) / downsample), 
    math.floor(delayest / downsample), minutter = minutter, searchbuf = 0)

  utt_starts[0] = searchbuf
  nutter = utt_starts.shape[0]
  utt_ends[nutter - 1] = reflen - searchbuf

  for utt_num in range(1, nutter):
    this_start = utt_starts[utt_num]
    last_end = utt_ends[utt_num - 1]
    count = math.floor((this_start + last_end) / 2)
    utt_starts[utt_num] = count
    utt_ends[utt_num - 1] = count

  this_start = utt_starts[0] * downsample + utt_delay[0]
  if this_start < bufsamp:
    count = searchbuf + math.floor((downsample - 1 - utt_delay[0]) / downsample)
    utt_starts[0] = count

  last_end = utt_ends[nutter - 1] * downsample + utt_delay[nutter - 1]
  if last_end > (deglen - bufsamp):
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
  sr_mod = 'wb' if sr == 16000 else 'nb'
  downsample = 32 if sr_mod == 'nb' else 64 
  align_nfft = 512 if sr_mod == 'nb' else 1024
  window = 0.5 * (1 - np.cos((2 * np.pi * np.arange(align_nfft)) / align_nfft))

  uttsearch_starts, uttsearch_ends = id_searchwindows(
    ref_vad, math.floor(reflen / downsample), 
    math.floor((deglen - delayest) / downsample), math.floor(delayest / downsample))
  nutter = uttsearch_starts.shape[0]

  utt_delayest = np.zeros(50)
  utt_delay = np.zeros(50)
  utt_delayconf = np.zeros(50)

  for utt_id in range(nutter):
    utt_delayest[utt_id] = align.frame_align(
      ref_logvad, uttsearch_starts[utt_id], uttsearch_ends[utt_id], 
      deg_logvad, math.floor(deglen / downsample),
      math.floor(delayest / downsample)) * downsample + delayest

    _utt_delay, _utt_delayconf = align.time_align(
      ref_data, deg_data, deglen,
      int(uttsearch_starts[utt_id] * downsample),
      int(uttsearch_ends[utt_id] * downsample),
      utt_delayest[utt_id], window)
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
      best_ed1, best_d1, best_dc1, best_ed2, best_d2, best_dc2, best_bp = split_align(
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

        nutter += 1

        utt_delayest[utt_id] = best_ed1
        utt_delay[utt_id] = best_d1
        utt_delayconf[utt_id] = best_dc1

        utt_delayest[utt_id + 1] = best_ed2
        utt_delay[utt_id + 1] = best_d2
        utt_delayconf[utt_id + 1] = best_dc2

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

  delayest = align.frame_align_all(
    ref_logvad, math.floor(reflen / downsample),
    deg_logvad, math.floor(deglen / downsample))
  delayest *= downsample

  utt_starts, utt_ends, utt_delay, nutter = utterance_locate(
    ref_data, reflen, ref_vad, ref_logvad, 
    deg_data, deglen, deg_vad, deg_logvad, delayest, sr)

  ref_data = model_ref
  deg_data = model_deg

  pesq_mos = pesq_model.pesq_model(ref_data, reflen, deg_data, deglen, sr,
    utt_starts, utt_ends, utt_delay, nutter)

  if sr_mod == 'nb':
    mapped = 0.999 + 4. / (1. + np.exp(-1.4945 * pesq_mos + 4.6607))
    return pesq_mos, mapped

  else:
    mapped = 0.999 + 4. / (1. + np.exp(-1.3669 * pesq_mos + 3.8224))
    return mapped

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
    utt_ed1[bp] = align.frame_align(
      ref_logvad, _utt_start, utt_bps[bp], 
      deg_logvad, math.floor(deglen / downsample),
      math.floor(utt_delayest_1 / downsample)) * downsample + utt_delayest_1

    utt_ed2[bp] = align.frame_align(
      ref_logvad, utt_bps[bp], _utt_end, 
      deg_logvad, math.floor(deglen / downsample),
      math.floor(utt_delayest_1 / downsample)) * downsample + utt_delayest_1

  utt_dc1 = np.zeros(41) 
  utt_dc1[:n_bps] = -2.

  def _inner(startr, startd, h, hsum, utt_d1, utt_dc1):
    while (startd + align_nfft) <= deglen and \
      (startr + align_nfft) <= (utt_bps[bp] * downsample):
      x1 = ref_data[int(startr) : int(startr + align_nfft)] * window
      x2 = deg_data[int(startd) : int(startd + align_nfft)] * window

      x1, v_max = align.xcorr(x1, x2, align_nfft)
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
      
      x1, v_max = align.xcorr(x1, x2, align_nfft)
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
      best_bp = utt_bps[bp]

  return best_ed1, best_d1, best_dc1, best_ed2, best_d2, best_dc2, best_bp
