import composite

csig, cbak, covl, segsnr = composite.composite("sp09.wav", "enhanced_logmmse.wav")
print(csig, cbak, covl, segsnr)
