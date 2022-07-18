import composite
import soundfile

data1, sr1 = soundfile.read("sp09.wav")
data2, sr2 = soundfile.read("enhanced_logmmse.wav")
assert sr1 == sr2

csig, cbak, covl, segsnr = composite.composite(data1, data2, sr1)
print(csig, cbak, covl, segsnr)
