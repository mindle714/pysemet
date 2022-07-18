# pysemet

python implementation of MATLAB code composite.zip from https://ecs.utdallas.edu/loizou/speech/software.htm

## How to use

```
import composite
import soundfile

data1, sr1 = soundfile.read("sp09.wav")
data2, sr2 = soundfile.read("enhanced_logmmse.wav")
assert sr1 == sr2

csig, cbak, covl, segsnr = composite.composite(data1, data2, sr1)
print(csig, cbak, covl, segsnr)
```

## References

MATLAB code from [https://ecs.utdallas.edu/loizou/speech/software.htm](https://ecs.utdallas.edu/loizou/speech/software.htm)

```bib
@ARTICLE{4389058,
  author={Y. {Hu} and P. C. {Loizou}},
  journal={IEEE Transactions on Audio, Speech, and Language Processing}, 
  title={Evaluation of Objective Quality Measures for Speech Enhancement}, 
  year={2008},
  volume={16},
  number={1},
  pages={229-238},
}
```
