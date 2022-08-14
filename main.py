import composite
import soundfile
import pesq
import math

data1, sr1 = soundfile.read("sp09.wav")
data2, sr2 = soundfile.read("enhanced_logmmse.wav")
assert sr1 == sr2
print(sr1)

sr_mod = 'wb' if sr1 == 16000 else 'nb'
pesq_mos = pesq.pesq(sr1, data1, data2, mode=sr_mod)
# convert to raw pesq on narrow-band case
if sr_mod == 'nb':
  pesq_mos = (math.log((4./(pesq_mos - 0.999))-1.)-4.6607) / -1.4945
print(pesq_mos)

csig, cbak, covl, segsnr = composite.composite(data1, data2, sr1)
print(csig, cbak, covl, segsnr)
