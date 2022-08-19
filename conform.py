import composite
import soundfile
import pesq
import math
import os

tcs = [e.strip() for e in open("conform/voipref.bat", "r").readlines()]
for tc in tcs:
  if tc == "": continue

  toks = tc.split()
  assert len(toks) == 4
  assert "".join(toks[:2]) == "pesq+8000" 

  data1, sr1 = soundfile.read(os.path.join("conform", toks[2]))
  data2, sr2 = soundfile.read(os.path.join("conform", toks[3]))
  assert sr1 == sr2

  raw_mos, pesq_mos = pesq.pesq(data1, data2, sr=sr1)
  print(raw_mos, pesq_mos)

