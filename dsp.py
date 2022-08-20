import numpy as np

def iirsos(x, Nx, b0, b1, b2, a1, a2, z1, z2):
  xpos = 0

  if a1 != 0. or a2 != 0.:
    if b1 != 0. or b2 != 0.:
      while Nx > 0:
        Nx -= 1
        z0 = x[xpos] - a1 * z1 - a2 * z2
        x[xpos] = b0 * z0 + b1 * z1 + b2 * z2
        xpos += 1
        z2 = z1
        z1 = z0

    else:
      if b0 != 1.:
        while Nx > 0:
          Nx -= 1
          z0 = x[xpos] - a1 * z1 - a2 * z2
          x[xpos] = b0 * z0
          xpos += 1
          z2 = z1
          z1 = z0

      else:
        while Nx > 0:
          Nx -= 1
          z0 = x[xpos] - a1 * z1 - a2 * z2
          x[xpos] = z0
          xpos += 1
          z2 = z1
          z1 = z0

  else:
    if b1 != 0. or b2 != 0.:
      while Nx > 0:
        Nx -= 1
        z0 = x[xpos]
        x[xpos] = b0 * z0 + b1 * z1 + b2 * z2
        xpos += 1
        z2 = z1
        z1 = z0

    else:
      if b0 != 1.:
        while Nx > 0:
          Nx -= 1
          x[xpos] = b0 * x[xpos]
          xpos += 1

  return z1, z2

def iirfilt(h, Nsos, x, Nx):
  y = np.copy(x[:Nx])
  hpos = 0

  for C in range(Nsos):
    iirsos(y, Nx, h[hpos], h[hpos+1], h[hpos+2], h[hpos+3], h[hpos+4], 0, 0)
    hpos += 5

  return y


