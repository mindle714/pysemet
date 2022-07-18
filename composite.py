import soundfile
import numpy as np
import math

cent_freq = [
  50.0000, 120.000, 190.000, 260.000, 330.000, 
  400.000, 470.000, 540.000, 617.372, 703.378, 
  798.717, 904.128, 1020.38, 1148.30, 1288.72, 
  1442.54, 1610.70, 1794.16, 1993.93, 2211.08,
  2446.71, 2701.97, 2978.04, 3276.17, 3597.63]

bandwidth = [
  70.0000, 70.0000, 70.0000, 70.0000, 70.0000,
  70.0000, 70.0000, 77.3724, 86.0056, 95.3398,
  105.411, 116.256, 127.914, 140.423, 153.823,
  168.154, 183.457, 199.776, 217.153, 235.631,
  255.255, 276.072, 298.126, 321.465, 346.136]

def nextpow2(x):
  return 1 if x == 0 else 2**math.ceil(math.log2(x))

def wss(clean_speech, processed_speech, sr):
  assert len(clean_speech) == len(processed_speech)
  winlength = round(30 * sr / 1000)
  skiprate = math.floor(winlength / 4)
  max_freq = sr / 2
  num_crit = 25

  USE_FFT_SPECTRUM = True
  n_fft = nextpow2(2*winlength)
  n_fftby2 = n_fft//2
  Kmax = 20
  Klocmax = 1
  bw_min = bandwidth[0]
  min_factor = math.exp(-30.0 / (2.0 * 2.303))

  crit_filter = np.zeros((num_crit, n_fftby2))
  for i in range(num_crit):
    f0 = (cent_freq[i] / max_freq) * n_fftby2
    bw = (bandwidth[i] / max_freq) * n_fftby2
    norm_factor = math.log(bw_min) - math.log(bandwidth[i])
    crit_filter[i,:] = np.exp(-11 * (((np.arange(n_fftby2)-math.floor(f0))/bw)**2) + norm_factor)
    crit_filter[i,:] = crit_filter[i,:] * (crit_filter[i,:] > min_factor)

  num_frames = int(len(clean_speech) / skiprate - (winlength / skiprate))
  start = 0
  window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1,winlength+1)/(winlength+1)))

  distortion = np.zeros(num_frames)
  for frame_count in range(num_frames):
    clean_frame = clean_speech[start:start+winlength]
    processed_frame = processed_speech[start:start+winlength]
    clean_frame = clean_frame * window
    processed_frame = processed_frame * window

    # compute power spectrum of clean/processed
    if USE_FFT_SPECTRUM:
      clean_spec = np.abs(np.fft.fft(clean_frame, n_fft))**2
      processed_spec = np.abs(np.fft.fft(processed_frame, n_fft))**2

    else:
      a_vec = np.zeros(n_fft)
      a_vec[:11] = lpc(clean_frame, 10)
      clean_spec = 1.0 / (np.abs(np.fft.fft(a_vec, n_fft))**2)
      
      a_vec = np.zeros(n_fft)
      a_vec[:11] = lpc(processed_frame, 10)
      processed_spec = 1.0 / (np.abs(np.fft.fft(a_vec, n_fft))**2)

    # compute filterbank output energy (db)
    clean_energy = np.zeros(num_crit)
    processed_energy = np.zeros(num_crit)
    for i in range(num_crit):
      clean_energy[i] = np.sum(clean_spec[:n_fftby2] * crit_filter[i,:])
      processed_energy[i] = np.sum(processed_spec[:n_fftby2] * crit_filter[i,:])

    clean_energy = 10 * np.log10(np.maximum(clean_energy, 1e-10))
    processed_energy = 10 * np.log10(np.maximum(processed_energy, 1e-10))

    # compute spectral slope
    clean_slope = clean_energy[1:] - clean_energy[:-1]
    processed_slope = processed_energy[1:] - processed_energy[:-1]

    # nearest peak location in the spectra to each critical band
    clean_loc_peak = np.zeros(num_crit-1)
    processed_loc_peak = np.zeros(num_crit-1)
    for i in range(num_crit-1):
      if clean_slope[i] > 0:
        n = i
        while (n < (num_crit-1)) and (clean_slope[n] > 0):
          n += 1
        clean_loc_peak[i] = clean_energy[n-1]

      else:
        n = i
        while (n>=0) and (clean_slope[n] <= 0):
          n -= 1
        clean_loc_peak[i] = clean_energy[n+1]

      if processed_slope[i] > 0:
        n = i
        while (n < (num_crit-1)) and (processed_slope[n] > 0):
          n += 1
        processed_loc_peak[i] = processed_energy[n-1]

      else:
        n = i
        while (n>=0) and (processed_slope[n] <= 0):
          n -= 1
        processed_loc_peak[i] = processed_energy[n+1]

    # compute wss measure for this frame
    dBMax_clean = np.max(clean_energy)
    dBMax_processed = np.max(processed_energy)

    Wmax_clean = Kmax / (Kmax + dBMax_clean - clean_energy[:num_crit-1])
    Wlocmax_clean = Klocmax / (Klocmax + clean_loc_peak - clean_energy[:num_crit-1])
    W_clean = Wmax_clean * Wlocmax_clean

    Wmax_processed = Kmax / (Kmax + dBMax_processed - processed_energy[:num_crit-1])
    Wlocmax_processed = Klocmax / (Klocmax + processed_loc_peak - processed_energy[:num_crit-1])
    W_processed = Wmax_processed * Wlocmax_processed

    W = (W_clean + W_processed) / 2.
    distortion[frame_count] = np.sum(W * (clean_slope[:num_crit-1] - processed_slope[:num_crit-1])**2)
    distortion[frame_count] /= np.sum(W)

    start += skiprate

  return distortion

def composite(cleanFile, enhancedFile):
  alpha = 0.95
  eps = 2.2204e-16

  data1, Srate1 = soundfile.read(cleanFile)
  data2, Srate2 = soundfile.read(enhancedFile)
  assert Srate1 == Srate2

  _len = min(data1.shape[0], data2.shape[0])
  data1 = data1[:_len] + eps
  data2 = data2[:_len] + eps

  wss_dist_vec = wss(data1, data2, Srate1)
